if args.sandwich_computation == "parallel":
# TODO I think this wont work now that I reshuffled the for loops around for implementing Higher
# DataParallel should be fine and we do actually want to share the same data across all models. But would need multi-GPU setup to check it out, it does not help on Single GPU

if epoch == 0:
    logger.log(f"Computing parallel sandwich forward pass at epoch = {epoch}")
# Prepare the multi-path samples in advance for Parallel Sandwich
if all_archs is not None:
    sandwich_archs = [random.sample(all_archs, 1)[0] for _ in range(args.sandwich)]
else:
    sandwich_archs = [arch_sampler.sample(mode="random", candidate_num = 1)[0] for _ in range(args.sandwich)]
network.zero_grad()
network.set_cal_mode('sandwich', sandwich_cells = sandwich_archs)
network.logits_only = True

if args.sandwich is not None and args.sandwich > 1:
    parallel_model = nn.DataParallel(network, device_ids = [0 for _ in range(args.sandwich)])
parallel_inputs = base_inputs.repeat((args.sandwich, 1, 1, 1))
parallel_targets = base_targets.repeat(args.sandwich)

all_logits = parallel_model(parallel_inputs)
parallel_loss = criterion(all_logits, parallel_targets)/args.sandwich
parallel_loss.backward()
split_logits = torch.split(all_logits, base_inputs.shape[0], dim=0)

network.logits_only = False
elif args.sandwich_computation == "parallel_custom":
# TODO probably useless. Does not provide any speedup due to only a single CUDA context being active on the GPU at a time even though the jobs are queued asynchronously
network.zero_grad()
all_logits = []
all_base_inputs = [deepcopy(base_inputs) for _ in range(args.sandwich)]
all_models = [deepcopy(network) for _ in range(args.sandwich)]
for sandwich_idx, sandwich_arch in enumerate(sandwich_archs):
    cur_model = all_models[sandwich_idx]
    cur_model.set_cal_mode('dynamic', sandwich_arch)
    _, logits = cur_model(base_inputs)
    all_logits.append(logits)
all_losses = [criterion(logits, base_targets) * (1 if args.sandwich is None else 1/args.sandwich) for logits in all_logits]
for loss in all_losses:
    loss.backward()
split_logits = all_logits