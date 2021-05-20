import torch
import time
import numpy as np

def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner

    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter

        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            torch.autograd.grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner

def hvp(dtrain_dw, vec, model):
    # Useful for Hessian-vector products in CG
    val = gather_flat_grad(torch.autograd.grad(dtrain_dw, model.parameters(),
                                grad_outputs=vec.view(-1), retain_graph=True))
    return val.view(1, -1, 1)

def gather_flat_grad(loss_grad):
    #cnt = 0
    #for g in loss_grad:
    #    g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
    #    cnt = 1
    return torch.cat([p.reshape(-1) for p in loss_grad]) #g_vector

def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-4, atol=0.0, maxiter=10, verbose=True):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves a batch of matrix linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.
    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

    if verbose:
        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)
        print("%03s | %010s %06s" % ("it", torch.max(residual_norm - stopping_matrix), "it/s"))

    optimal = False
    start = time.perf_counter()
    cur_error = 1e-8
    epsilon = 1e-2
    for k in range(1, maxiter + 1):
        # epsilon = cur_error ** 3  # 1e-8

        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator < epsilon / 2] = epsilon  # epsilon
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator < epsilon / 2] = epsilon
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        cur_error = torch.max(residual_norm - stopping_matrix)
        if verbose:
            print("%03d | %8.6e %4.2f" %
                  (k, cur_error,
                   1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))

    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info

def hyper_step(model, train_loader, val_loader, criterion, arch_params, arch_params_real, elementary_lr, max_iter=None, algo ="cg"):
    """Estimate the hypergradient, and take an update with it.

    :param get_hyper_train:  A function which returns the hyperparameters we want to tune.
    :param model:  A function which returns the elementary parameters we want to tune.
    :param val_loss_func:  A function which takes input x and output y, then returns the scalar valued loss.
    :param val_loader: A generator for input x, output y tuples.
    :param d_train_loss_d_w:  The derivative of the training loss with respect to elementary parameters.
    :param hyper_optimizer: The optimizer which updates the hyperparameters.
    :return: The scalar valued validation loss, the hyperparameter norm, and the hypergradient norm.
    """
    zero_hypergrad(arch_params)
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in arch_params)
    print(f"num_weights : {num_weights}, num_hypers : {num_hypers}")
    # d_train_loss_d_w = gather_flat_grad(d_train_loss_d_w)  # TODO: COmmented this out!
    d_train_loss_d_w = torch.zeros(num_weights).cuda()
    model.train(), model.zero_grad()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to('cuda'), y.to('cuda')
        _, logits = model(x)
        train_loss = criterion(logits, y)
        # train_loss, _ = train_loss_func(x, y)
        model.zero_grad()
        d_train_loss_d_w += gather_flat_grad(torch.autograd.grad(train_loss, model.parameters(), create_graph=True))
        break
    model.zero_grad()

    # Compute gradients of the validation loss w.r.t. the weights/hypers
    d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).cuda(), torch.zeros(num_hypers).cuda()
    model.train(), model.zero_grad()
    for batch_idx, (x, y) in enumerate(val_loader):
        x, y = x.to('cuda'), y.to('cuda')
        _, logits = model(x)
        val_loss = criterion(logits, y)
        # val_loss = val_loss_func(x, y)
        model.zero_grad()
        d_val_loss_d_theta += gather_flat_grad(torch.autograd.grad(val_loss, model.parameters())) # TODO should there be retain_graph=True? It was there for the reweighting net
        break

    # Initialize the preconditioner and counter
    preconditioner = d_val_loss_d_theta
    if algo == "neumann":
        preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr,
                                                            max_iter, model)
    elif algo == "cg":
        preconditioner, _ = cg_batch(lambda vec: hvp(d_train_loss_d_w, vec, model), d_val_loss_d_theta.view(1, -1, 1),
                                        maxiter=max_iter)
    # compute d / d lambda (partial Lv / partial w * partial Lt / partial w)
    # = (partial Lv / partial w * partial^2 Lt / (partial w partial lambda))
    indirect_grad = gather_flat_grad(
        torch.autograd.grad(d_train_loss_d_w, arch_params, grad_outputs=preconditioner.view(-1)))
    hypergrad = direct_grad + indirect_grad

    zero_hypergrad(arch_params)
    store_hypergrad(arch_params_real, hypergrad)
    print(hypergrad)
    return val_loss, hypergrad

def zero_hypergrad(arch_params):
    current_index = 0
    for p in arch_params:
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0
        current_index += p_num_params
        
def store_hypergrad(arch_params, total_dval_dlambda):
    current_index = 0
    for p in arch_params:
        p_num_params = np.prod(p.shape)
        p.grad = total_dval_dlambda[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params