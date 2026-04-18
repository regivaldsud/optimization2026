# ============================================================
# Lista T02 - Otimizacao
# Resolucao em Julia
# ============================================================

# Uncomment once if needed:
# using Pkg
# Pkg.add(["Optim", "NLsolve", "Plots", "JuMP", "HiGHS"])

# Avoid GUI plotting issues when running from terminal/script mode.
ENV["GKSwstype"] = "100"

using LinearAlgebra
using Printf
using Optim
using NLsolve
using Plots
using JuMP
using HiGHS

default(size = (850, 500), linewidth = 2)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function logsumexp2(a, b)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

function softmax2(a, b)
    m = max(a, b)
    ea = exp(a - m)
    eb = exp(b - m)
    den = ea + eb
    return ea / den, eb / den
end

function summarize_run(label, f, x_star, traj)
    println(label)
    println("  solution   = ", x_star)
    println("  f(solution)= ", f(x_star))
    println("  iterations = ", length(traj) - 1)
end

function gradient_descent_fixed(grad, x0; step = 0.1, tol = 1e-8, maxiter = 10_000)
    x = copy(x0)
    trajectory = [copy(x)]
    for _ in 1:maxiter
        g = grad(x)
        if norm(g) <= tol
            return x, trajectory
        end
        x = x .- step .* g
        push!(trajectory, copy(x))
    end
    return x, trajectory
end

function newton_fixed(grad, hess, x0; tol = 1e-8, maxiter = 100)
    x = copy(x0)
    trajectory = [copy(x)]
    for _ in 1:maxiter
        g = grad(x)
        if norm(g) <= tol
            return x, trajectory
        end
        x = x .- (hess(x) \ g)
        push!(trajectory, copy(x))
    end
    return x, trajectory
end

function finite_difference_hessian(grad, x; h = 1e-5)
    n = length(x)
    H = zeros(n, n)
    g0 = grad(x)
    for i in 1:n
        ei = zeros(n)
        ei[i] = h
        H[:, i] = (grad(x .+ ei) .- g0) ./ h
    end
    return 0.5 .* (H + H')
end

function in_domain(x)
    return (x[1] + x[2] > 0.0) && (x[3] > 0.0)
end

function ensure_domain_step(x, dx; beta = 0.5)
    t = 1.0
    while !in_domain(x .+ t .* dx)
        t *= beta
    end
    return t
end

function backtracking_line_search(f, grad, x, dx; alpha = 0.4, beta = 0.5)
    t = ensure_domain_step(x, dx; beta = beta)
    fx = f(x)
    gx = grad(x)
    while f(x .+ t .* dx) > fx + alpha * t * dot(gx, dx)
        t *= beta
    end
    return t
end

function gradient_descent_backtracking(f, grad, x0; alpha = 0.4, beta = 0.5, eps = 1e-5, maxiter = 10_000)
    x = copy(x0)
    history = [copy(x)]
    for k in 1:maxiter
        g = grad(x)
        if norm(g) <= eps
            return x, history, k - 1, f(x)
        end
        dx = -g
        t = backtracking_line_search(f, grad, x, dx; alpha = alpha, beta = beta)
        x = x .+ t .* dx
        push!(history, copy(x))
    end
    error("Gradient descent did not converge in the maximum number of iterations.")
end

function newton_backtracking(f, grad, x0; alpha = 0.4, beta = 0.5, eps = 1e-5, h = 1e-5, maxiter = 200)
    x = copy(x0)
    history = [copy(x)]
    for k in 1:maxiter
        g = grad(x)
        H = finite_difference_hessian(grad, x; h = h)
        Hg = H \ g
        lambda2 = dot(g, Hg)
        if lambda2 / 2 <= eps
            return x, history, k - 1, f(x), lambda2
        end
        dx = -Hg
        t = backtracking_line_search(f, grad, x, dx; alpha = alpha, beta = beta)
        x = x .+ t .* dx
        push!(history, copy(x))
    end
    error("Newton's method did not converge in the maximum number of iterations.")
end

function bfgs_backtracking(f, grad, x0; alpha = 0.4, beta = 0.5, eps = 1e-5, maxiter = 2000, ys_tol = 1e-4, rho_cap = 1e4)
    x = copy(x0)
    n = length(x)
    H = Matrix{Float64}(I, n, n)
    history = [copy(x)]
    for k in 1:maxiter
        g = grad(x)
        if norm(g) <= eps
            return x, history, H, k - 1, f(x)
        end
        p = -H * g
        t = backtracking_line_search(f, grad, x, p; alpha = alpha, beta = beta)
        s = t .* p
        x_new = x .+ s
        g_new = grad(x_new)
        y = g_new .- g
        ys = dot(y, s)
        rho = ys <= ys_tol ? rho_cap : 1 / ys
        I_n = Matrix{Float64}(I, n, n)
        V = I_n - rho .* (s * y')
        H = V * H * V' + rho .* (s * s')
        x = x_new
        push!(history, copy(x))
    end
    error("BFGS did not converge in the maximum number of iterations.")
end

# ------------------------------------------------------------
# Section 1 - FONC example
# ------------------------------------------------------------

function run_fonc_example()
    println("\n============================================================")
    println("FONC example")
    println("============================================================")

    f_example(x1, x2) = (x1 - 3)^2 + (x2 + 4)^2

    x1_grid = range(-1.0, 7.0; length = 120)
    x2_grid = range(-8.0, 0.0; length = 120)
    Z = [f_example(a, b) for b in x2_grid, a in x1_grid]

    p1 = surface(x1_grid, x2_grid, Z;
        xlabel = "x1", ylabel = "x2", zlabel = "f(x)",
        title = "Surface of (x1 - 3)^2 + (x2 + 4)^2")

    p2 = contour(x1_grid, x2_grid, Z;
        xlabel = "x1", ylabel = "x2",
        title = "Level curves",
        fill = true)
    scatter!(p2, [3.0], [-4.0]; color = :red, label = "minimum")

    display(plot(p1, p2; layout = (1, 2)))
end

# ------------------------------------------------------------
# Section 2 - Quartic + plots
# ------------------------------------------------------------

function run_quartic_problem()
    println("\n============================================================")
    println("Solving the FONC with Optim and NLsolve")
    println("============================================================")

    f_quartic(x) = (x[1] - 1)^4 + (x[2] - 2)^4

    function grad_quartic!(g, x)
        g[1] = 4 * (x[1] - 1)^3
        g[2] = 4 * (x[2] - 2)^3
    end

    function fonc_quartic!(F, x)
        F[1] = 4 * (x[1] - 1)^3
        F[2] = 4 * (x[2] - 2)^3
    end

    x0 = [0.0, 0.0]
    res_optim = optimize(f_quartic, grad_quartic!, x0, BFGS())
    res_nlsolve = nlsolve(fonc_quartic!, x0)

    println("Optim minimizer = ", Optim.minimizer(res_optim))
    println("Optim minimum   = ", Optim.minimum(res_optim))
    println("NLsolve root    = ", res_nlsolve.zero)

    f_quad_plot(x1, x2) = (x1 - 3)^2 + (x2 - 2)^2
    x1_grid = range(-1.0, 7.0; length = 120)
    x2_grid = range(-1.0, 5.0; length = 120)
    Z = [f_quad_plot(a, b) for b in x2_grid, a in x1_grid]

    p1 = surface(x1_grid, x2_grid, Z;
        xlabel = "x1", ylabel = "x2", zlabel = "f(x)",
        title = "Surface of (x1 - 3)^2 + (x2 - 2)^2")

    p2 = contour(x1_grid, x2_grid, Z;
        xlabel = "x1", ylabel = "x2",
        title = "Level curves",
        fill = true)
    scatter!(p2, [3.0], [2.0]; color = :red, label = "minimum")

    display(plot(p1, p2; layout = (1, 2)))
end

# ------------------------------------------------------------
# Section 3 - Quadratic methods
# ------------------------------------------------------------

function run_quadratic_methods()
    println("\n============================================================")
    println("Gradient descent and Newton for quadratic functions")
    println("============================================================")

    f1(x) = (x[1] - 3)^2 + (x[2] - 2)^2
    grad1(x) = [2 * (x[1] - 3), 2 * (x[2] - 2)]
    hess1(x) = [2.0 0.0; 0.0 2.0]
    x01 = [1.0, 1.0]
    x_gd1, traj_gd1 = gradient_descent_fixed(grad1, x01; step = 0.5)
    x_nt1, traj_nt1 = newton_fixed(grad1, hess1, x01)
    summarize_run("f(x) = (x1 - 3)^2 + (x2 - 2)^2 with gradient descent", f1, x_gd1, traj_gd1)
    summarize_run("f(x) = (x1 - 3)^2 + (x2 - 2)^2 with Newton", f1, x_nt1, traj_nt1)

    f2(x) = 10 * (x[1] - 3)^2 + 2 * (x[2] - 2)^2
    grad2(x) = [20 * (x[1] - 3), 4 * (x[2] - 2)]
    hess2(x) = [20.0 0.0; 0.0 4.0]
    x02 = [1.0, 1.0]
    x_gd2, traj_gd2 = gradient_descent_fixed(grad2, x02; step = 0.05)
    x_nt2, traj_nt2 = newton_fixed(grad2, hess2, x02)
    summarize_run("f(x) = 10(x1 - 3)^2 + 2(x2 - 2)^2 with gradient descent", f2, x_gd2, traj_gd2)
    summarize_run("f(x) = 10(x1 - 3)^2 + 2(x2 - 2)^2 with Newton", f2, x_nt2, traj_nt2)

    f3(x) = x[1]^2 + x[2]^2
    grad3(x) = [2 * x[1], 2 * x[2]]
    hess3(x) = [2.0 0.0; 0.0 2.0]
    x03 = [-2.0, 2.0]
    x_gd3, traj_gd3 = gradient_descent_fixed(grad3, x03; step = 0.5)
    x_nt3, traj_nt3 = newton_fixed(grad3, hess3, x03)
    summarize_run("f(x) = x1^2 + x2^2 with gradient descent", f3, x_gd3, traj_gd3)
    summarize_run("f(x) = x1^2 + x2^2 with Newton", f3, x_nt3, traj_nt3)

    f4(x) = x[1]^2 + 100 * x[2]^2
    grad4(x) = [2 * x[1], 200 * x[2]]
    hess4(x) = [2.0 0.0; 0.0 200.0]
    x04 = [-2.0, 2.0]
    x_gd4, traj_gd4 = gradient_descent_fixed(grad4, x04; step = 0.005)
    x_nt4, traj_nt4 = newton_fixed(grad4, hess4, x04)
    summarize_run("f(x) = x1^2 + 100x2^2 with gradient descent", f4, x_gd4, traj_gd4)
    summarize_run("f(x) = x1^2 + 100x2^2 with Newton", f4, x_nt4, traj_nt4)

    g_change(v) = v[1]^2 + v[2]^2
    grad_change(v) = [2 * v[1], 2 * v[2]]
    hess_change(v) = [2.0 0.0; 0.0 2.0]
    v0 = [-2.0, 20.0]
    v_gd, traj_v_gd = gradient_descent_fixed(grad_change, v0; step = 0.5)
    v_nt, traj_v_nt = newton_fixed(grad_change, hess_change, v0)

    println("After the change x2 = y2 / 10:")
    println("  GD solution in y-space = ", v_gd, " -> x-space = ", [v_gd[1], v_gd[2] / 10])
    println("  NT solution in y-space = ", v_nt, " -> x-space = ", [v_nt[1], v_nt[2] / 10])
    println("  GD iterations in y-space = ", length(traj_v_gd) - 1)
    println("  NT iterations in y-space = ", length(traj_v_nt) - 1)
end

# ------------------------------------------------------------
# Section 4 - Main convex function
# ------------------------------------------------------------

function f_main(x)
    in_domain(x) || error("Point outside dom f")
    x1, x2, x3 = x
    s = x1 + x2
    return x3 * logsumexp2(x1 / x3, x2 / x3) + (x3 - 2)^2 + exp(1 / s)
end

function grad_main(x)
    in_domain(x) || error("Point outside dom f")
    x1, x2, x3 = x
    s = x1 + x2
    p1, p2 = softmax2(x1 / x3, x2 / x3)
    lse = logsumexp2(x1 / x3, x2 / x3)
    common = exp(1 / s) / s^2
    return [
        p1 - common,
        p2 - common,
        lse - (x1 * p1 + x2 * p2) / x3 + 2 * (x3 - 2)
    ]
end

function run_gradient_descent_main()
    println("\n============================================================")
    println("Gradient descent with backtracking line search")
    println("============================================================")

    alpha_bt = 0.4
    beta_bt = 0.5
    eps_stop = 1e-5
    x_start = [3.0, 4.0, 5.0]

    dx0 = -grad_main(x_start)
    println("Initial domain-safe step = ", ensure_domain_step(x_start, dx0; beta = beta_bt))
    println("Initial backtracking step = ", backtracking_line_search(f_main, grad_main, x_start, dx0; alpha = alpha_bt, beta = beta_bt))

    x_gd, hist_gd, it_gd, f_gd = gradient_descent_backtracking(
        f_main, grad_main, x_start;
        alpha = alpha_bt, beta = beta_bt, eps = eps_stop
    )

    println("Gradient descent result")
    println("  x*         = ", x_gd)
    println("  f(x*)      = ", f_gd)
    println("  iterations = ", it_gd)
    println("  ||grad||   = ", norm(grad_main(x_gd)))
end

function run_newton_main()
    println("\n============================================================")
    println("Newton's method with backtracking")
    println("============================================================")

    alpha_bt = 0.4
    beta_bt = 0.5
    eps_stop = 1e-5
    h_fd = 1e-5
    x_start = [3.0, 4.0, 5.0]

    H0 = finite_difference_hessian(grad_main, x_start; h = h_fd)
    dx0 = -(H0 \ grad_main(x_start))
    println("Initial backtracking step = ", backtracking_line_search(f_main, grad_main, x_start, dx0; alpha = alpha_bt, beta = beta_bt))

    x_nt, hist_nt, it_nt, f_nt, lambda2_nt = newton_backtracking(
        f_main, grad_main, x_start;
        alpha = alpha_bt, beta = beta_bt, eps = eps_stop, h = h_fd
    )

    println("Newton result")
    println("  x*           = ", x_nt)
    println("  f(x*)        = ", f_nt)
    println("  iterations   = ", it_nt)
    println("  lambda^2 / 2 = ", lambda2_nt / 2)
    println("  ||grad||     = ", norm(grad_main(x_nt)))
    println("  Hessian at x* = ")
    println(finite_difference_hessian(grad_main, x_nt; h = h_fd))
end

function run_bfgs_main()
    println("\n============================================================")
    println("BFGS with backtracking")
    println("============================================================")

    alpha_bt = 0.4
    beta_bt = 0.5
    eps_stop = 1e-5
    x_start = [3.0, 4.0, 5.0]

    H0 = Matrix{Float64}(I, 3, 3)
    dx0 = -H0 * grad_main(x_start)
    println("Initial backtracking step = ", backtracking_line_search(f_main, grad_main, x_start, dx0; alpha = alpha_bt, beta = beta_bt))

    x_bfgs, hist_bfgs, H_bfgs, it_bfgs, f_bfgs = bfgs_backtracking(
        f_main, grad_main, x_start;
        alpha = alpha_bt, beta = beta_bt, eps = eps_stop
    )

    println("BFGS result")
    println("  x*         = ", x_bfgs)
    println("  f(x*)      = ", f_bfgs)
    println("  iterations = ", it_bfgs)
    println("  ||grad||   = ", norm(grad_main(x_bfgs)))
    println("  Final inverse-Hessian approximation = ")
    println(H_bfgs)
end

# ------------------------------------------------------------
# Section 5 - CSTR LP with JuMP
# ------------------------------------------------------------

function run_cstr_lp()
    println("\n============================================================")
    println("Optimal control for CSTR via LP/JuMP")
    println("============================================================")

    A = [
        0.2681  -0.00338  -0.00728;
        9.7030   0.3279  -25.4400;
        0.0      0.0       1.0
    ]

    B = [
        -0.00537   0.1655;
         1.29700  97.9100;
         0.0      -6.6370
    ]

    C = Matrix{Float64}(I, 3, 3)
    N = 3
    x0 = [-0.03, 0.0, 0.3]
    xmin = [-0.05, -5.0, -0.5]
    xmax = [ 0.05,  5.0,  0.5]
    umin = [-10.0, -0.05]
    umax = [ 10.0,  0.05]

    nx, nu = size(B)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, xmin[i] <= x[i = 1:nx, k = 0:N] <= xmax[i])
    @variable(model, umin[j] <= u[j = 1:nu, k = 0:N-1] <= umax[j])
    @variable(model, z[1:2, 0:N] >= 0)

    @constraint(model, [i in 1:nx], x[i, 0] == x0[i])

    @constraint(model, [i in 1:nx, k in 0:N-1],
        x[i, k + 1] == sum(A[i, j] * x[j, k] for j in 1:nx) + sum(B[i, j] * u[j, k] for j in 1:nu)
    )

    @constraint(model, [k in 0:N], -z[1, k] <= x[1, k])
    @constraint(model, [k in 0:N],  x[1, k] <= z[1, k])
    @constraint(model, [k in 0:N], -z[2, k] <= x[3, k])
    @constraint(model, [k in 0:N],  x[3, k] <= z[2, k])

    @objective(model, Min, sum(z[1, k] + z[2, k] for k in 0:N))

    optimize!(model)

    println("termination status = ", termination_status(model))
    println("objective value    = ", objective_value(model))

    X = [value(x[i, k]) for i in 1:nx, k in 0:N]
    U = [value(u[j, k]) for j in 1:nu, k in 0:N-1]

    println("Optimal states X = ")
    println(X)
    println("Optimal inputs U = ")
    println(U)

    t_state = collect(0:N)
    t_input = collect(0:N-1)

    p_y = plot(t_state, X[1, :];
        marker = :circle,
        label = "y1 = c - cs",
        xlabel = "k",
        ylabel = "controlled variables",
        title = "Controlled variables")
    plot!(p_y, t_state, X[3, :]; marker = :square, label = "y3 = h - hs")
    hline!(p_y, [0.0]; linestyle = :dash, color = :black, label = "setpoint")

    p_u = plot(t_input, U[1, :];
        seriestype = :steppost,
        marker = :circle,
        label = "u1 = Tc - Tcs",
        xlabel = "k",
        ylabel = "inputs",
        title = "Manipulated variables")
    plot!(p_u, t_input, U[2, :];
        seriestype = :steppost,
        marker = :square,
        label = "u2 = F - Fs")

    display(plot(p_y, p_u; layout = (2, 1), size = (850, 750)))

    return model, X, U, A, B, C
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main()
    run_fonc_example()
    run_quartic_problem()
    run_quadratic_methods()
    run_gradient_descent_main()
    run_newton_main()
    run_bfgs_main()
    run_cstr_lp()
end

main()
