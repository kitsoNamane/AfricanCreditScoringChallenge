module AfricanCreditScoringChallenge

using CSV
using DataFrames
using Statistics

export readdataset, linearscaling, logscaling, zscorescaling, clipscaling, compute_cost_logistic, custom_model

# Data Wrangling

function readdataset(filepath::String)
    df = CSV.read(filepath, DataFrame)
    df = select!(df, Not([:ID, :customer_id, :country_id, :tbl_loan_id, :lender_id, :disbursement_date, :due_date]))
    df = transform!(df, :loan_type => ByRow(x -> parse(Int64, x[end])) => :loan_type)
    df = transform!(df, :New_versus_Repeat => ByRow(y -> if (y == "Repeat Loan")
        1
    else
        0
    end) => :New_versus_Repeat)
    Matrix(select(df, Not(:target))), df.target
end

function linearscaling(x::Vector)
    x_min = minimum(x)
    x_max = maximum(x)

    (x .- x_min) ./ (x_max - x_min)
end

function logscaling(x::Vector)
    log1p(x)
end

function zscorescaling(x::Vector)
    x_mean = mean(x)
    x_std = std(x)

    (x .- x_mean) ./ x_std
end

function clipscaling(x::Vector, min, max)
    for i in eachindex(x)
        if (x[i] > max)
            x[i] = max
        elseif (x[i] < min)
            x[i] = min
        end
    end
end

# Machine Learning Algorithms

function sigmoid(z)
    1 ./ (1 .+ exp.(-z))
end

function custom_model(w::Vector, x::Matrix, b)
    z = x * w .+ b
    sigmoid(z)
end

function compute_cost_logistic(w::Vector, b, x::Matrix, y::Vector)
    f_wb = custom_model(w, x, b)
    cost = -y .* log.(f_wb) .- (1 .- y) .* log.(1 .- f_wb)
    sum(cost) / size(x)[1]
end

function compute_gradient_logistic(w::Vector, b, x::Matrix, y::Vector)
    m = size(x)[1]

    f_wb = custom_model(w, x, b)
    err = f_wb .- y
    dj_dw = sum.(transpose(err) * x) ./ m
    dj_dw[1, :], sum(err) / m
end

function gradient_descent(w::Vector, b, alpha, x::Matrix, y::Vector, num_iters::Int64)
    for _ in 1:num_iters
        dj_dw, dj_db = compute_gradient_logistic(w, b, x, y)
        w = w .- alpha * dj_dw
        b = b - alpha * dj_db
    end

    return w, b
end

function threshhold(x, th)
    if x <= th
        return 0
    else
        return 1
    end
end

function predict(w::Vector, b, x::Matrix)
    threshhold.(custom_model(w, x, b))
end

end # module AfricanCreditScoringChallenge
