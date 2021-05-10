def sample_new_arch(network, algo, arch_sampler, sandwich_archs, all_archs, base_inputs, base_targets, arch_overview, loss_threshold, args):
# Need to sample a new architecture (considering it as a meta-batch dimension)

    sampling_done = False # Used for GreedyNAS online search space pruning - might have to resample many times until we find an architecture below the required threshold
    lowest_loss_arch = None
    lowest_loss = 10000
    while not sampling_done: # TODO the sampling_done should be useful for like online sampling with rejections maybe
        if algo == 'setn':
            sampled_arch = network.dync_genotype(True)
            network.set_cal_mode('dynamic', sampled_arch)
        elif algo == 'gdas':
            network.set_cal_mode('gdas', None)
            sampled_arch = network.genotype
        elif algo.startswith('darts'):
            network.set_cal_mode('joint', None)
            sampled_arch = network.genotype

        elif "random_" in algo and len(parsed_algo) > 1 and ("perf" in algo or "size" in algo):
            if args.search_space_paper == "nats-bench":
                sampled_arch = arch_sampler.sample()[0]
                network.set_cal_mode('dynamic', sampled_arch)
            else:
                network.set_cal_mode('urs')
        # elif "random" in algo and args.evenly_split is not None: # TODO should just sample outside of the function and pass it in as all_archs?
        #   sampled_arch = arch_sampler.sample(mode="evenly_split", candidate_num = args.eval_candidate_num)[0]
        #   network.set_cal_mode('dynamic', sampled_arch)

        elif "random" in algo and args.sandwich is not None and args.sandwich > 1 and args.sandwich_computation == "parallel":
            assert args.sandwich_mode != "quartiles", "Not implemented yet"
            sampled_arch = sandwich_archs[outer_iter]
            network.set_cal_mode('dynamic', sampled_arch)

        elif "random" in algo and args.sandwich is not None and args.sandwich > 1 and args.sandwich_mode == "quartiles":
            if args.search_space_paper == "nats-bench":
                assert args.sandwich == 4 # 4 corresponds to using quartiles
                if step == 0:
                    logger.log(f"Sampling from the Sandwich branch with sandwich={args.sandwich} and sandwich_mode={args.sandwich_mode}")
                    sampled_archs = arch_sampler.sample(mode = "quartiles", subset = all_archs, candidate_num=args.sandwich) # Always samples 4 new archs but then we pick the one from the right quartile
                    sampled_arch = sampled_archs[outer_iter] # Pick the corresponding quartile architecture for this iteration
                    network.set_cal_mode('dynamic', sampled_arch)
            else:
                network.set_cal_mode('urs')
        elif "random_" in algo and "grad" in algo:
            network.set_cal_mode('urs')
        elif algo == 'random': # NOTE the original branch needs to be last so that it is fall-through for all the special 'random' branches
            if supernets_decomposition or all_archs is not None or arch_groups_brackets is not None:
                if all_archs is not None:
                    sampled_arch = random.sample(all_archs, 1)[0]
                    network.set_cal_mode('dynamic', sampled_arch)
                else:
                    if args.search_space_paper == "nats-bench":
                        sampled_arch = arch_sampler.sample(mode="random")[0]
                        network.set_cal_mode('dynamic', sampled_arch)
                    else:
                        network.set_cal_mode('urs', None)
            else:
                network.set_cal_mode('urs', None)
        elif algo == 'enas':
            with torch.no_grad():
                network.controller.eval()
                _, _, sampled_arch = network.controller()
            network.set_cal_mode('dynamic', sampled_arch)
        else:
            raise ValueError('Invalid algo name : {:}'.format(algo))
        if loss_threshold is not None:
            with torch.no_grad():
                _, logits = network(base_inputs)
                base_loss = criterion(logits, base_targets) * (1 if args.sandwich is None else 1/args.sandwich)
                if base_loss.item() < lowest_loss:
                    lowest_loss = base_loss.item()
                    lowest_loss_arch = sampled_arch
                if base_loss.item() < loss_threshold:
                    sampling_done = True
        else:
            sampling_done = True
        if sampling_done:
            arch_overview["cur_arch"] = sampled_arch
            arch_overview["all_archs"].append(sampled_arch)
            arch_overview["all_cur_archs"].append(sampled_arch)
    return sampled_arch

def format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args):

    base_inputs, arch_inputs = base_inputs.cuda(non_blocking=True), arch_inputs.cuda(non_blocking=True)
    base_targets, arch_targets = base_targets.cuda(non_blocking=True), arch_targets.cuda(non_blocking=True)
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = [base_inputs], [base_targets], [arch_inputs], [arch_targets]
    for extra_step in range(inner_steps-1):
        if args.inner_steps_same_batch:
            all_base_inputs.append(base_inputs)
            all_base_targets.append(base_targets)
            all_arch_inputs.append(arch_inputs)
            all_arch_targets.append(arch_targets)
            continue # If using the same batch, we should not try to query the search_loader_iter for more samples
        try:
            extra_base_inputs, extra_base_targets, extra_arch_inputs, extra_arch_targets = next(search_loader_iter)
        except:
            continue
        extra_base_inputs, extra_arch_inputs = extra_base_inputs.cuda(non_blocking=True), extra_arch_inputs.cuda(non_blocking=True)
        extra_base_targets, extra_arch_targets = extra_base_targets.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        all_base_inputs.append(extra_base_inputs)
        all_base_targets.append(extra_base_targets)
        all_arch_inputs.append(extra_arch_inputs)
        all_arch_targets.append(extra_arch_targets)

    return all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets


def update_brackets(supernet_train_stats_by_arch, supernet_train_stats, supernet_train_stats_avgmeters, arch_groups_brackets, arch_overview, items, all_brackets, sampled_arch, args):
    if type(arch_groups_brackets) is dict:
        cur_bracket = arch_groups_brackets[arch_overview["cur_arch"].tostr()]
        for key, val in items:
            supernet_train_stats_by_arch[sampled_arch.tostr()][key].append(val)
            for bracket in all_brackets:
                if bracket == cur_bracket:
                    supernet_train_stats[key]["sup"+str(cur_bracket)].append(val)
                    supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(cur_bracket)].update(val)
                    supernet_train_stats[key+"AVG"]["sup"+str(cur_bracket)].append(supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(cur_bracket)].avg)
                else:
                    item_to_add = supernet_train_stats[key]["sup"+str(bracket)][-1] if len(supernet_train_stats[key]["sup"+str(bracket)]) > 0 else 3.14159
                    supernet_train_stats[key]["sup"+str(bracket)].append(item_to_add)
                    avg_to_add = supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].avg if supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].avg > 0 else 3.14159
                    supernet_train_stats[key+"AVG"]["sup"+str(bracket)].append(avg_to_add)