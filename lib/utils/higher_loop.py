import torch
from copy import deepcopy


def hypergrad_outer(
    args,
    fnetwork,
    criterion,
    arch_targets,
    arch_inputs,
    all_arch_inputs,
    all_arch_targets,
    all_base_inputs,
    all_base_targets,
    sotl,
    inner_step,
    inner_rollouts,
    first_order_grad_for_free_cond,
    first_order_grad_concurrently_cond,
    monkeypatch_higher_grads_cond,
    zero_arch_grads_lambda,
    step,
    epoch,
    logger,
    inner_steps,
):
    meta_grads = []
    if args.meta_algo in ["reptile", "metaprox"]:
        inner_rollouts.append(deepcopy(fnetwork.state_dict()))
    elif args.meta_algo:
        if args.higher_method.startswith("val"):
            if args.higher_order == "second":
                _, logits = fnetwork(
                    arch_inputs, params=fnetwork.parameters(time=inner_step)
                )
                arch_loss = [criterion(logits, arch_targets)]
                meta_grad = torch.autograd.grad(
                    sum(arch_loss), fnetwork.parameters(time=0), allow_unused=True
                )
                meta_grads.append(meta_grad)
            elif args.higher_order == "first":
                if not (
                    first_order_grad_for_free_cond or first_order_grad_concurrently_cond
                ):  # Computing the val grads concurrently allows to avoid gradient tracking in Higher
                    if args.higher_method == "val":
                        all_logits = [
                            fnetwork(arch_inputs, params=fnetwork.parameters(time=i))[1]
                            for i in range(0, inner_steps)
                        ]
                        arch_loss = [
                            criterion(all_logits[i], arch_targets)
                            for i in range(len(all_logits))
                        ]
                    elif args.higher_method == "val_multiple":
                        all_logits = [
                            fnetwork(
                                all_arch_inputs[i], params=fnetwork.parameters(time=i)
                            )[1]
                            for i in range(0, inner_steps)
                        ]
                        arch_loss = [
                            criterion(all_logits[i], all_arch_targets[i])
                            for i in range(len(all_logits))
                        ]
                    all_grads = [
                        torch.autograd.grad(arch_loss[i], fnetwork.parameters(time=i))
                        for i in range(0, inner_steps)
                    ]
                    if step == 0 and epoch < 2:
                        logger.log(
                            f"Reductioning all_grads (len={len(all_grads)} with reduction={args.higher_reduction}, inner_steps={inner_steps}"
                        )
                    if args.higher_reduction == "sum":

                        fo_grad = [sum(grads) for grads in zip(*all_grads)]
                    elif args.higher_reduction == "mean":
                        fo_grad = [
                            sum(grads) / inner_steps for grads in zip(*all_grads)
                        ]
                    meta_grads.append(fo_grad)
                else:
                    pass

        elif args.higher_method == "sotl":
            if args.higher_order == "second":
                meta_grad = torch.autograd.grad(
                    sum(sotl), fnetwork.parameters(time=0), allow_unused=True
                )
                meta_grads.append(meta_grad)

            elif args.higher_order == "first":
                if not (
                    first_order_grad_for_free_cond or first_order_grad_concurrently_cond
                ):  # TODO I think the for_free branch puts each individual FO grad into meta_grads but here we put only average - though shouldnt really make a difference I think since we just sum over them either now or later?
                    all_logits = [
                        fnetwork(
                            all_base_inputs[i], params=fnetwork.parameters(time=i)
                        )[1]
                        for i in range(0, inner_steps)
                    ]
                    arch_loss = [
                        criterion(all_logits[i], all_base_targets[i])
                        for i in range(len(all_logits))
                    ]
                    all_grads = [
                        torch.autograd.grad(arch_loss[i], fnetwork.parameters(time=i))
                        for i in range(0, inner_steps)
                    ]
                    if step == 0 and epoch < 2:
                        logger.log(
                            f"Reductioning all_grads (len={len(all_grads)} with reduction={args.higher_reduction}, inner_steps={inner_steps}"
                        )
                        print(f"Grads sample before: {all_grads[0]}")
                    with torch.no_grad():
                        if args.higher_reduction == "sum":
                            fo_grad = [sum(grads) for grads in zip(*all_grads)]
                        elif args.higher_reduction == "mean":
                            fo_grad = [
                                sum(grads) / inner_steps for grads in zip(*all_grads)
                            ]
                    if step == 0:
                        print(f"Grads sample after: {fo_grad[0]}")
                    meta_grads.append(fo_grad)
                elif step == 0 and monkeypatch_higher_grads_cond:
                    all_logits = [
                        fnetwork(
                            all_base_inputs[i], params=fnetwork.parameters(time=i)
                        )[1]
                        for i in range(0, inner_steps)
                    ]
                    arch_loss = [
                        criterion(all_logits[i], all_base_targets[i])
                        for i in range(len(all_logits))
                    ]
                    all_grads = [
                        torch.autograd.grad(arch_loss[i], fnetwork.parameters(time=i))
                        for i in range(0, inner_steps)
                    ]
                    assert torch.all_close(
                        zero_arch_grads_lambda(all_grads[0]), meta_grads[0]
                    )
                    logger.log(
                        f"Correctnes of first-order gradients was checked! Samples:"
                    )
                    print(all_grads[0][0])
                    print(meta_grads[0][0])
                else:
                    pass
    return meta_grads, inner_rollouts
