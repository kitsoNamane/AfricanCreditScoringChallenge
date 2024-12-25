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
    y = select(df, :target => :target)
    Matrix(select(df, Not(:target))), Matrix(y)
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

end # module AfricanCreditScoringChallenge
