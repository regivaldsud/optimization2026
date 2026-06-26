# =====================================================================
#  TRABALHO 02 - Otimização 2026 (PPGEE/UFAM)
#  Análise computacional de 3 problemas de scheduling com Julia + HiGHS
#    1) Machine Scheduling   (1 | r_j | sum T_j)  -> min tardiness total
#    2) Job Shop Scheduling  (J || C_max)         -> min makespan
#    3) Flow Shop (permut.)  (F | prmu | C_max)   -> min makespan
#  Todos os modelos são MILP disjuntivos (big-M) resolvidos com HiGHS.
# =====================================================================
using JuMP, HiGHS, JSON, Printf, Dates

const ROOT      = normpath(joinpath(@__DIR__, ".."))
const DATA      = joinpath(ROOT, "data")
const RESULTS   = joinpath(ROOT, "results")
isdir(RESULTS) || mkpath(RESULTS)

# Configuração LIVRE do solver HiGHS (permitido pelo enunciado)
const TIME_LIMIT_MACHINE  = 30.0     # s por instância
const TIME_LIMIT_JOBSHOP  = 60.0
const TIME_LIMIT_FLOWSHOP = 60.0
const MIP_REL_GAP         = 1e-4     # 0.01%
const MIP_FEAS_TOL        = 1e-6

make_solver(tlim) = optimizer_with_attributes(HiGHS.Optimizer,
    "time_limit"            => tlim,
    "mip_rel_gap"           => MIP_REL_GAP,
    "primal_feasibility_tolerance" => MIP_FEAS_TOL,
    "dual_feasibility_tolerance"   => MIP_FEAS_TOL,
    "output_flag"           => false,
)

status_str(m) = string(termination_status(m))

function rel_gap(m)
    try
        return relative_gap(m)
    catch
        return NaN
    end
end

# ---------------------------------------------------------------------
# 1) MACHINE SCHEDULING  -  1 | r_j | sum T_j
# ---------------------------------------------------------------------
function solve_machine(file)
    d = JSON.parsefile(file)
    n   = Int(d["n"])
    r   = Float64.(d["release"])
    p   = Float64.(d["duration"])
    due = Float64.(d["due"])
    jobs = String.(d["jobs"])
    H = maximum(r) + sum(p)          # horizonte / big-M

    model = Model(make_solver(TIME_LIMIT_MACHINE))
    @variable(model, s[1:n] >= 0)                 # início
    @variable(model, T[1:n] >= 0)                 # tardiness
    @variable(model, x[1:n, 1:n], Bin)            # x[i,j]=1 se i antes de j
    @constraint(model, [j=1:n], s[j] >= r[j])
    @constraint(model, [j=1:n], T[j] >= s[j] + p[j] - due[j])
    for i in 1:n, j in 1:n
        if i < j
            @constraint(model, s[j] >= s[i] + p[i] - H * (1 - x[i, j]))
            @constraint(model, s[i] >= s[j] + p[j] - H * x[i, j])
        end
    end
    @objective(model, Min, sum(T))
    t0 = time(); optimize!(model); elapsed = time() - t0

    sched = []
    if has_values(model)
        for j in 1:n
            st = value(s[j]); ft = st + p[j]
            push!(sched, Dict("job"=>jobs[j], "start"=>round(st,digits=3),
                "finish"=>round(ft,digits=3), "due"=>due[j],
                "tardiness"=>round(max(0, ft-due[j]),digits=3)))
        end
        sort!(sched, by = z -> z["start"])
    end
    return Dict(
        "instance"=>d["name"], "n"=>n,
        "objective"=> has_values(model) ? round(objective_value(model),digits=4) : nothing,
        "bound"=> round(objective_bound(model),digits=4),
        "gap"=> round(rel_gap(model),digits=6),
        "status"=> status_str(model), "runtime"=> round(elapsed,digits=3),
        "schedule"=>sched)
end

# ---------------------------------------------------------------------
# 2) JOB SHOP  -  J || C_max     (formato JSPLIB)
# ---------------------------------------------------------------------
function parse_jsplib(file)
    lines = [strip(l) for l in readlines(file) if !startswith(strip(l), "#") && strip(l) != ""]
    hdr = split(lines[1])
    nj, nm = parse(Int, hdr[1]), parse(Int, hdr[2])
    # ops[j] = vetor de (machine0idx, duração)
    ops = Vector{Vector{Tuple{Int,Int}}}()
    for j in 1:nj
        toks = parse.(Int, split(lines[1+j]))
        v = Tuple{Int,Int}[]
        for k in 1:2:length(toks)
            push!(v, (toks[k], toks[k+1]))   # máquina indexada em 0
        end
        push!(ops, v)
    end
    return nj, nm, ops
end

function solve_jobshop(file, name)
    nj, nm, ops = parse_jsplib(file)
    bigM = sum(sum(p for (_, p) in ops[j]) for j in 1:nj)
    model = Model(make_solver(TIME_LIMIT_JOBSHOP))
    nop = [length(ops[j]) for j in 1:nj]
    @variable(model, st[j=1:nj, o=1:nop[j]] >= 0)   # início da operação
    @variable(model, Cmax >= 0)
    # precedência dentro do job
    for j in 1:nj, o in 1:(nop[j]-1)
        @constraint(model, st[j,o+1] >= st[j,o] + ops[j][o][2])
    end
    # makespan
    for j in 1:nj
        @constraint(model, Cmax >= st[j,nop[j]] + ops[j][nop[j]][2])
    end
    # operações por máquina
    bymachine = Dict{Int, Vector{Tuple{Int,Int}}}()
    for j in 1:nj, o in 1:nop[j]
        m = ops[j][o][1]
        push!(get!(bymachine, m, Tuple{Int,Int}[]), (j, o))
    end
    # disjunção por máquina
    z = Dict{Tuple{Int,Int,Int,Int}, VariableRef}()
    for (m, lst) in bymachine
        for a in 1:length(lst), b in (a+1):length(lst)
            (j1,o1) = lst[a]; (j2,o2) = lst[b]
            zz = @variable(model, binary=true)
            z[(j1,o1,j2,o2)] = zz
            p1 = ops[j1][o1][2]; p2 = ops[j2][o2][2]
            @constraint(model, st[j2,o2] >= st[j1,o1] + p1 - bigM*(1-zz))
            @constraint(model, st[j1,o1] >= st[j2,o2] + p2 - bigM*zz)
        end
    end
    @objective(model, Min, Cmax)
    t0 = time(); optimize!(model); elapsed = time() - t0

    sched = []
    if has_values(model)
        for j in 1:nj, o in 1:nop[j]
            s0 = value(st[j,o]); pp = ops[j][o][2]
            push!(sched, Dict("job"=>j, "op"=>o, "machine"=>ops[j][o][1],
                "start"=>round(s0,digits=3), "finish"=>round(s0+pp,digits=3)))
        end
    end
    return Dict("instance"=>name, "jobs"=>nj, "machines"=>nm,
        "objective"=> has_values(model) ? round(objective_value(model),digits=4) : nothing,
        "bound"=> round(objective_bound(model),digits=4),
        "gap"=> round(rel_gap(model),digits=6),
        "status"=>status_str(model), "runtime"=>round(elapsed,digits=3),
        "schedule"=>sched)
end

# ---------------------------------------------------------------------
# 3) FLOW SHOP de permutação  -  F | prmu | C_max  (CSV jobs x máquinas)
# ---------------------------------------------------------------------
function parse_fssp(file)
    rows = [split(strip(l), ",") for l in readlines(file) if strip(l) != ""]
    header = rows[1]
    machines = header[2:end]
    jobnames = String[]; P = Vector{Vector{Float64}}()
    for r in rows[2:end]
        push!(jobnames, r[1])
        push!(P, parse.(Float64, r[2:end]))
    end
    nj = length(jobnames); nm = length(machines)
    return nj, nm, jobnames, P
end

function solve_flowshop(file, name)
    nj, nm, jobnames, P = parse_fssp(file)
    p(j,k) = P[j][k]
    bigM = sum(sum(P[j]) for j in 1:nj)
    model = Model(make_solver(TIME_LIMIT_FLOWSHOP))
    @variable(model, C[1:nj, 1:nm] >= 0)            # conclusão job j na máquina k
    @variable(model, x[1:nj, 1:nj], Bin)            # x[i,j]=1 se i precede j (mesma ordem em todas máquinas)
    @variable(model, Cmax >= 0)
    # primeira máquina
    @constraint(model, [j=1:nj], C[j,1] >= p(j,1))
    # precedência de máquinas dentro do job
    for j in 1:nj, k in 2:nm
        @constraint(model, C[j,k] >= C[j,k-1] + p(j,k))
    end
    # disjunção de sequência (mesma permutação em todas as máquinas)
    for i in 1:nj, j in 1:nj
        if i < j
            for k in 1:nm
                @constraint(model, C[j,k] >= C[i,k] + p(j,k) - bigM*(1-x[i,j]))
                @constraint(model, C[i,k] >= C[j,k] + p(i,k) - bigM*x[i,j])
            end
        end
    end
    @constraint(model, [j=1:nj], Cmax >= C[j,nm])
    @objective(model, Min, Cmax)
    t0 = time(); optimize!(model); elapsed = time() - t0

    sched = []
    if has_values(model)
        for j in 1:nj, k in 1:nm
            cf = value(C[j,k])
            push!(sched, Dict("job"=>jobnames[j], "machine"=>k,
                "start"=>round(cf-p(j,k),digits=3), "finish"=>round(cf,digits=3)))
        end
    end
    return Dict("instance"=>name, "jobs"=>nj, "machines"=>nm,
        "objective"=> has_values(model) ? round(objective_value(model),digits=4) : nothing,
        "bound"=> round(objective_bound(model),digits=4),
        "gap"=> round(rel_gap(model),digits=6),
        "status"=>status_str(model), "runtime"=>round(elapsed,digits=3),
        "schedule"=>sched)
end

# ---------------------------------------------------------------------
#  RUNNER
# ---------------------------------------------------------------------
function run_machine()
    dir = joinpath(DATA, "machine")
    files = sort(filter(f -> endswith(f, ".json"), readdir(dir)))
    res = []
    for f in files
        @printf("  [machine] %-16s ... ", f)
        r = solve_machine(joinpath(dir, f))
        @printf("obj=%s gap=%s %s (%.1fs)\n", r["objective"], r["gap"], r["status"], r["runtime"])
        push!(res, r)
    end
    return res
end

function run_jobshop()
    meta = JSON.parsefile(joinpath(DATA, "jobshop", "instances.json"))
    res = []
    for inst in meta
        name = inst["name"]
        f = joinpath(DATA, "jobshop", "instances", name)
        @printf("  [jobshop] %-8s (%dx%d, opt=%s) ... ", name, inst["jobs"], inst["machines"], inst["optimum"])
        r = solve_jobshop(f, name)
        r["optimum"] = inst["optimum"]
        @printf("obj=%s bound=%s gap=%s %s (%.1fs)\n", r["objective"], r["bound"], r["gap"], r["status"], r["runtime"])
        push!(res, r)
    end
    return res
end

function run_flowshop()
    dir = joinpath(DATA, "flowshop")
    files = sort(filter(f -> endswith(f, ".csv"), readdir(dir)))
    res = []
    for f in files
        @printf("  [flowshop] %-20s ... ", f)
        r = solve_flowshop(joinpath(dir, f), replace(f, ".csv"=>""))
        @printf("obj=%s gap=%s %s (%.1fs)\n", r["objective"], r["gap"], r["status"], r["runtime"])
        push!(res, r)
    end
    return res
end

println("=== TRABALHO 02 :: Machine Scheduling ===")
machine = run_machine()
println("=== TRABALHO 02 :: Job Shop ===")
jobshop = run_jobshop()
println("=== TRABALHO 02 :: Flow Shop ===")
flowshop = run_flowshop()

all_results = Dict(
    "config" => Dict(
        "solver" => "HiGHS",
        "time_limit_machine" => TIME_LIMIT_MACHINE,
        "time_limit_jobshop" => TIME_LIMIT_JOBSHOP,
        "time_limit_flowshop" => TIME_LIMIT_FLOWSHOP,
        "mip_rel_gap" => MIP_REL_GAP,
        "mip_feas_tol" => MIP_FEAS_TOL,
        "julia_version" => string(VERSION),
        "timestamp" => string(now()),
    ),
    "machine" => machine,
    "jobshop" => jobshop,
    "flowshop" => flowshop,
)
open(joinpath(RESULTS, "results.json"), "w") do io
    JSON.print(io, all_results, 2)
end
println("\nResultados salvos em results/results.json")
