using LinearAlgebra
using Printf

const OUTDIR = joinpath(@__DIR__, "resultados")
mkpath(OUTDIR)

function norm2(x)
    return x isa Number ? abs(x) : norm(x)
end

function fmt(x::Number)
    return @sprintf("%.12g", x)
end

function fmt(x::AbstractVector)
    return "[" * join(fmt.(x), ", ") * "]"
end

function table_scalar(hist)
    lines = ["| k | x | F(x) |", "|---:|---:|---:|"]
    for (k, x, fx) in hist
        push!(lines, "| $k | $(fmt(x)) | $(fmt(fx)) |")
    end
    return join(lines, "\n")
end

function table_vector(hist)
    lines = ["| k | x | ||F(x)|| |", "|---:|:---|---:|"]
    for (k, x, fx) in hist
        push!(lines, "| $k | $(fmt(x)) | $(fmt(norm(fx))) |")
    end
    return join(lines, "\n")
end

function newton1(f, df, x0; eps=1e-15, maxiter=50)
    x = float(x0)
    hist = Tuple{Int, Float64, Float64}[]
    for k in 0:maxiter
        fx = f(x)
        push!(hist, (k, x, fx))
        abs(fx) <= eps && return x, hist, :converged
        dfx = df(x)
        abs(dfx) <= Base.eps(float(x)) && return x, hist, :singular_derivative
        x -= fx / dfx
    end
    return x, hist, :maxiter
end

function newtonn(F, J, x0; eps=1e-15, maxiter=50)
    x = float.(copy(x0))
    hist = Tuple{Int, Vector{Float64}, Vector{Float64}}[]
    for k in 0:maxiter
        fx = F(x)
        push!(hist, (k, copy(x), fx))
        norm(fx) <= eps && return x, hist, :converged
        x -= J(x) \ fx
    end
    return x, hist, :maxiter
end

function finite_newton1(f, x0; tau=1e-7, eps=1e-15, maxiter=50)
    x = float(x0)
    hist = Tuple{Int, Float64, Float64}[]
    for k in 0:maxiter
        fx = f(x)
        push!(hist, (k, x, fx))
        abs(fx) <= eps && return x, hist, :converged
        a = (f(x + tau) - fx) / tau
        abs(a) <= Base.eps(float(x)) && return x, hist, :singular_slope
        x -= fx / a
    end
    return x, hist, :maxiter
end

function secant1(f, x0, a0; eps=1e-15, maxiter=50)
    x = float(x0)
    a = float(a0)
    hist = Tuple{Int, Float64, Float64}[]
    for k in 0:maxiter
        fx = f(x)
        push!(hist, (k, x, fx))
        abs(fx) <= eps && return x, hist, :converged
        abs(a) <= Base.eps(float(x)) && return x, hist, :singular_slope
        s = -fx / a
        xnew = x + s
        fnew = f(xnew)
        a = (fnew - fx) / s
        x = xnew
    end
    return x, hist, :maxiter
end

function finite_jacobian(F, x, tau)
    fx = F(x)
    n = length(x)
    J = zeros(length(fx), n)
    for j in 1:n
        xp = copy(x)
        xp[j] += tau
        J[:, j] = (F(xp) - fx) / tau
    end
    return J
end

function finite_newtonn(F, x0; tau=1e-7, eps=1e-15, maxiter=50)
    x = float.(copy(x0))
    hist = Tuple{Int, Vector{Float64}, Vector{Float64}}[]
    for k in 0:maxiter
        fx = F(x)
        push!(hist, (k, copy(x), fx))
        norm(fx) <= eps && return x, hist, :converged
        x -= finite_jacobian(F, x, tau) \ fx
    end
    return x, hist, :maxiter
end

function broyden(F, x0; tau=1e-7, eps=1e-15, maxiter=50)
    x = float.(copy(x0))
    B = finite_jacobian(F, x, tau)
    hist = Tuple{Int, Vector{Float64}, Vector{Float64}}[]
    for k in 0:maxiter
        fx = F(x)
        push!(hist, (k, copy(x), fx))
        norm(fx) <= eps && return x, hist, :converged
        s = -(B \ fx)
        xnew = x + s
        y = F(xnew) - fx
        B += ((y - B * s) * s') / dot(s, s)
        x = xnew
    end
    return x, hist, :maxiter
end

function assert_pd(Q)
    issymmetric(Q) || error("A matriz precisa ser simetrica.")
    cholesky(Symmetric(Q))
    return eigvals(Symmetric(Q))
end

function quadratic_direct(Q, b)
    assert_pd(Q)
    return -(Q \ b)
end

function conjugate_gradient_quadratic(Q, b, x0; eps=1e-12, maxiter=100)
    assert_pd(Q)
    x = float.(copy(x0))
    g = Q * x + b
    d = -g
    hist = Tuple{Int, Vector{Float64}, Float64}[(0, copy(x), norm(g))]
    for k in 1:maxiter
        alpha = dot(g, g) / dot(d, Q * d)
        x += alpha * d
        gnew = Q * x + b
        push!(hist, (k, copy(x), norm(gnew)))
        norm(gnew) <= eps && return x, hist, :converged
        beta = dot(gnew, gnew) / dot(g, g)
        d = -gnew + beta * d
        g = gnew
    end
    return x, hist, :maxiter
end

function newton_minimize(f, grad, hess, x0; eps=1e-12, maxiter=50)
    x = float.(copy(x0))
    hist = Tuple{Int, Vector{Float64}, Float64, Float64}[]
    for k in 0:maxiter
        g = grad(x)
        push!(hist, (k, copy(x), f(x), norm(g)))
        norm(g) <= eps && return x, hist, :converged
        H = hess(x)
        cholesky(Symmetric(H))
        x -= H \ g
    end
    return x, hist, :maxiter
end

function rosenbrock(x)
    return sum(100.0 * (x[i + 1] - x[i]^2)^2 + (1.0 - x[i])^2 for i in 1:length(x)-1)
end

function rosenbrock_grad(x)
    n = length(x)
    g = zeros(n)
    for i in 1:n-1
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i]^2) - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * (x[i + 1] - x[i]^2)
    end
    return g
end

function rosenbrock_hess(x)
    n = length(x)
    H = zeros(n, n)
    for i in 1:n-1
        H[i, i] += 1200.0 * x[i]^2 - 400.0 * x[i + 1] + 2.0
        H[i, i + 1] += -400.0 * x[i]
        H[i + 1, i] += -400.0 * x[i]
        H[i + 1, i + 1] += 200.0
    end
    return H
end

function exact_line_initialization(h; delta=1e-2, maxiter=60)
    a = 0.0
    b = delta
    h(b) >= h(a) && return (0.0, b)
    for _ in 1:maxiter
        c = 2.0 * b
        h(c) >= h(b) && return (a, c)
        a, b = b, c
    end
    return (a, b)
end

function quadratic_interpolation_line_search(h; eps=1e-12, maxiter=60)
    a, c = exact_line_initialization(h)
    b = (a + c) / 2.0
    for _ in 1:maxiter
        ha, hb, hc = h(a), h(b), h(c)
        denom = (b - a) * (hb - hc) - (b - c) * (hb - ha)
        abs(denom) <= eps && break
        x = b - 0.5 * ((b - a)^2 * (hb - hc) - (b - c)^2 * (hb - ha)) / denom
        abs(x - b) <= eps && return x
        if x < b
            h(x) < hb ? (c, b = b, x) : (a = x)
        else
            h(x) < hb ? (a, b = b, x) : (c = x)
        end
    end
    return b
end

function backtracking(f, grad, x, d; beta=0.5, sigma=1e-4)
    alpha = 1.0
    fx = f(x)
    gd = dot(grad(x), d)
    while f(x + alpha * d) > fx + sigma * alpha * gd
        alpha *= beta
    end
    return alpha
end

function steepest_descent(f, grad, x0; eps=1e-8, maxiter=20_000)
    x = float.(copy(x0))
    hist = Tuple{Int, Vector{Float64}, Float64, Float64}[]
    for k in 0:maxiter
        g = grad(x)
        push!(hist, (k, copy(x), f(x), norm(g)))
        norm(g) <= eps && return x, hist, :converged
        d = -g
        alpha = backtracking(f, grad, x, d)
        x += alpha * d
    end
    return x, hist, :maxiter
end

function bfgs(f, grad, x0; eps=1e-8, maxiter=10_000)
    x = float.(copy(x0))
    n = length(x)
    H = Matrix{Float64}(I, n, n)
    hist = Tuple{Int, Vector{Float64}, Float64, Float64}[]
    for k in 0:maxiter
        g = grad(x)
        push!(hist, (k, copy(x), f(x), norm(g)))
        norm(g) <= eps && return x, hist, :converged
        d = -H * g
        dot(g, d) >= 0 && (d = -g; H = Matrix{Float64}(I, n, n))
        alpha = backtracking(f, grad, x, d)
        xnew = x + alpha * d
        s = xnew - x
        y = grad(xnew) - g
        ys = dot(y, s)
        if ys > 1e-14
            rho = 1.0 / ys
            V = Matrix{Float64}(I, n, n) - rho * s * y'
            H = V * H * V' + rho * s * s'
        end
        x = xnew
    end
    return x, hist, :maxiter
end

function write_svg_newton(hist, path)
    width, height = 760, 420
    margin = 55
    xs = [k for (k, _, _) in hist]
    ys = [abs(fx) for (_, _, fx) in hist]
    ys = max.(ys, 1e-18)
    ly = log10.(ys)
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ly)
    ymax == ymin && (ymax += 1)
    px(k) = margin + (width - 2margin) * (k - xmin) / max(1, xmax - xmin)
    py(v) = height - margin - (height - 2margin) * (log10(max(abs(v), 1e-18)) - ymin) / (ymax - ymin)
    points = join(["$(px(k)),$(py(fx))" for (k, _, fx) in hist], " ")
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/><line x1="$margin" y1="$(height-margin)" x2="$(width-margin)" y2="$(height-margin)" stroke="black"/><line x1="$margin" y1="$margin" x2="$margin" y2="$(height-margin)" stroke="black"/>""")
        println(io, """<text x="$(width/2)" y="28" text-anchor="middle" font-family="Arial" font-size="18">Newton: iteracoes x |F(x)| em escala log10</text>""")
        println(io, """<text x="$(width/2)" y="$(height-12)" text-anchor="middle" font-family="Arial" font-size="13">iteracao</text>""")
        println(io, """<text x="18" y="$(height/2)" transform="rotate(-90 18,$(height/2))" text-anchor="middle" font-family="Arial" font-size="13">log10 |F(x)|</text>""")
        println(io, """<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="$points"/>""")
        for (k, _, fx) in hist
            println(io, """<circle cx="$(px(k))" cy="$(py(fx))" r="4" fill="#d62728"/>""")
        end
        println(io, "</svg>")
    end
end

function summarize_last(hist)
    last(hist)
end

function main()
    md = String[]
    push!(md, "# Atividades- Aula 03- Regivaldo Araújo\n")

    F1(x) = x^2 - 2
    dF1(x) = 2x
    x, hist, status = newton1(F1, dF1, 2.0, eps=1e-15)
    write_svg_newton(hist, joinpath(OUTDIR, "newton_x2menos2.svg"))
    push!(md, "## S03-01 - Newton\n")
    push!(md, "### F(x)=x^2-2, x0=2\nStatus: `$status`; raiz aproximada: `$(fmt(x))`; F(x): `$(fmt(F1(x)))`.\n")
    push!(md, table_scalar(hist) * "\n")

    F2(x) = x - sin(x)
    dF2(x) = 1 - cos(x)
    x, hist, status = newton1(F2, dF2, 1.0, eps=1e-15)
    push!(md, "### F(x)=x-sin(x), x0=1\nStatus: `$status`; raiz aproximada: `$(fmt(x))`; F(x): `$(fmt(F2(x)))`.\n")
    push!(md, table_scalar(hist) * "\n")

    F3(x) = atan(x)
    dF3(x) = 1 / (1 + x^2)
    x, hist, status = newton1(F3, dF3, 1.5, eps=1e-15, maxiter=10)
    push!(md, "### F(x)=atan(x), x0=1.5, maxiter=10\nStatus: `$status`; ultimo x: `$(fmt(x))`; F(x): `$(fmt(F3(x)))`.\n")
    push!(md, table_scalar(hist) * "\n")

    Fv1(x) = [(x[1] + 1)^2 + x[2]^2 - 2, exp(x[1]) + x[2]^3 - 2]
    Jv1(x) = [2(x[1] + 1) 2x[2]; exp(x[1]) 3x[2]^2]
    x, hist, status = newtonn(Fv1, Jv1, [1.0, 1.0], eps=1e-15)
    push!(md, "### Newton n variaveis - exemplo 7.11\nStatus: `$status`; solucao: `$(fmt(x))`; ||F(x)||: `$(fmt(norm(Fv1(x))))`.\n")
    push!(md, table_vector(hist) * "\n")

    Fv2(x) = [x[1]^3 - 3x[1] * x[2]^2 - 1, x[2]^3 - 3x[1]^2 * x[2]]
    Jv2(x) = [3x[1]^2 - 3x[2]^2 -6x[1]*x[2]; -6x[1]*x[2] 3x[2]^2 - 3x[1]^2]
    for x0 in ([1.0, 1.0], [-1.0, -1.0], [0.0, 1.0])
        x, hist, status = newtonn(Fv2, Jv2, x0, eps=1e-15)
        push!(md, "### Newton n variaveis - polinomio, x0=$(fmt(x0))\nStatus: `$status`; solucao: `$(fmt(x))`; ||F(x)||: `$(fmt(norm(Fv2(x))))`.\n")
        push!(md, table_vector(hist) * "\n")
    end

    push!(md, "## S03-02 - Quasi-Newton\n")
    for tau in (1e-7, 0.1)
        x, hist, status = finite_newton1(F1, 2.0, tau=tau, eps=1e-15)
        push!(md, "### Diferencas finitas 1D, tau=$(tau)\nStatus: `$status`; raiz aproximada: `$(fmt(x))`; F(x): `$(fmt(F1(x)))`.\n")
        push!(md, table_scalar(hist) * "\n")
    end
    x, hist, status = secant1(F1, 2.0, 1.0, eps=1e-15)
    push!(md, "### Secante 1D, a0=1\nStatus: `$status`; raiz aproximada: `$(fmt(x))`; F(x): `$(fmt(F1(x)))`.\n")
    push!(md, table_scalar(hist) * "\n")
    for tau in (1e-7, 0.1)
        x, hist, status = finite_newtonn(Fv1, [1.0, 1.0], tau=tau, eps=1e-15)
        push!(md, "### Diferencas finitas n variaveis, tau=$(tau)\nStatus: `$status`; solucao: `$(fmt(x))`; ||F(x)||: `$(fmt(norm(Fv1(x))))`.\n")
        push!(md, table_vector(hist) * "\n")
    end
    x, hist, status = broyden(Fv1, [1.0, 1.0], eps=1e-15)
    push!(md, "### Secante/Broyden n variaveis\nStatus: `$status`; solucao: `$(fmt(x))`; ||F(x)||: `$(fmt(norm(Fv1(x))))`.\n")
    push!(md, table_vector(hist) * "\n")

    push!(md, "## S03-03 - Problemas quadraticos\n")
    Q = [1.0 1 1 1; 1 2 2 2; 1 2 3 3; 1 2 3 4]
    b = [-4.0, -7, -9, -10]
    vals = assert_pd(Q)
    xdir = quadratic_direct(Q, b)
    push!(md, "Autovalores de Q: `$(fmt(vals))`, todos positivos. Solucao direta: `$(fmt(xdir))`.\n")
    xcg, histcg, status = conjugate_gradient_quadratic(Q, b, [5.0, 5, 5, 5])
    push!(md, "Gradiente conjugado: status `$status`; solucao `$(fmt(xcg))`; ||Qx+b|| `$(fmt(norm(Q*xcg+b)))`.\n")
    badQ = [1.0 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
    try
        assert_pd(badQ)
        push!(md, "Matriz nao positiva: nenhum erro detectado.\n")
    catch err
        push!(md, "Matriz do teste nao e definida positiva: `$(sprint(showerror, err))`.\n")
    end

    push!(md, "## S03-04 - Metodo local de Newton\n")
    f58(x) = 0.5 * x[1]^2 + x[1] * cos(x[2])
    g58(x) = [x[1] + cos(x[2]), -x[1] * sin(x[2])]
    H58(x) = [1.0 -sin(x[2]); -sin(x[2]) -x[1] * cos(x[2])]
    try
        x, hist, status = newton_minimize(f58, g58, H58, [1.0, 1.0])
        push!(md, "Exemplo 5.8: status `$status`; ponto `$(fmt(x))`; f(x) `$(fmt(f58(x)))`; ||grad|| `$(fmt(norm(g58(x))))`.\n")
    catch err
        push!(md, "Exemplo 5.8 falhou porque a Hessiana nao e positiva definida em alguma iteracao: `$(sprint(showerror, err))`.\n")
    end
    x, hist, status = newton_minimize(rosenbrock, rosenbrock_grad, rosenbrock_hess, [-1.2, 1.0])
    push!(md, "Rosenbrock com Newton/modelo quadratico: status `$status`; ponto `$(fmt(x))`; f(x) `$(fmt(rosenbrock(x)))`; iteracoes `$(length(hist)-1)`.\n")

    push!(md, "## S03-05 - Busca linear e descida\n")
    h(t) = (2 + t) * cos(2 + t)
    interval = exact_line_initialization(h)
    alpha = quadratic_interpolation_line_search(h)
    push!(md, "Inicializacao de busca exata para h(t)=(2+t)cos(2+t): intervalo `$(fmt(collect(interval)))`; interpolacao quadratica alpha `$(fmt(alpha))`; h(alpha) `$(fmt(h(alpha)))`.\n")
    fq(x) = 0.5 * x[1]^2 + 4.5 * x[2]^2
    gq(x) = [x[1], 9x[2]]
    alphaq = backtracking(fq, gq, [9.0, 1.0], -gq([9.0, 1.0]))
    push!(md, "Busca linear no exemplo 11.2 a partir de [9,1]: alpha `$(fmt(alphaq))`.\n")
    x, hist, status = steepest_descent(rosenbrock, rosenbrock_grad, [-1.2, 1.0], eps=1e-6)
    push!(md, "Descida mais ingreme no Rosenbrock: status `$status`; ponto `$(fmt(x))`; f(x) `$(fmt(rosenbrock(x)))`; ||grad|| `$(fmt(norm(rosenbrock_grad(x))))`; iteracoes `$(length(hist)-1)`.\n")

    push!(md, "## S03-06 - BFGS\n")
    x, hist, status = bfgs(f58, g58, [1.0, 1.0], eps=1e-8)
    push!(md, "BFGS no exemplo 5.8: status `$status`; ponto `$(fmt(x))`; f(x) `$(fmt(f58(x)))`; ||grad|| `$(fmt(norm(g58(x))))`; iteracoes `$(length(hist)-1)`.\n")
    x, hist, status = bfgs(rosenbrock, rosenbrock_grad, [-1.2, 1.0], eps=1e-8)
    push!(md, "BFGS no Rosenbrock: status `$status`; ponto `$(fmt(x))`; f(x) `$(fmt(rosenbrock(x)))`; ||grad|| `$(fmt(norm(rosenbrock_grad(x))))`; iteracoes `$(length(hist)-1)`.\n")

    report = join(md, "\n")
    write(joinpath(OUTDIR, "RESULTADOS.md"), report)
    println(report)
    println("\nArquivos gerados:")
    println(joinpath(OUTDIR, "RESULTADOS.md"))
    println(joinpath(OUTDIR, "newton_x2menos2.svg"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
