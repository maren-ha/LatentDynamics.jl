"""
    mutable struct SMATestData

Struct to serve as a container for the SMArtCARE data, consisting of the following fields: 
    - `test`: name of the motor function test for which the data is collected
    - `xs`: vector of matrices (n_items x n_timepoints) of the item scores across time of
             the chosen test for each patient
    - `xs_baseline`: vector of vectors of baseline variable measurements for each patient
    - `tvals`: vector of vectors of follow-up time points for each patient
    - `ids`: vector of patient IDs
"""
mutable struct SMATestData
    test::String
    xs::Vector{Matrix{Float32}}
    xs_baseline::Vector{Vector{Float32}}
    tvals::Vector{Vector{Float32}}
    ids::Vector
end

function get_test_variables(test, colnames)
    if occursin("chop", test)
        test_vars = filter(x-> occursin("chop", x), colnames)
    elseif occursin("rulm", test)
        test_vars = filter(x-> occursin("rulm", x), colnames)
    elseif occursin("hfmse", test)
        test_vars = filter(x-> occursin("hfmse", x), colnames)
    elseif occursin("six", test)
        test_vars = filter(x -> occursin("distance", x), colnames)
    else
        error("invalid test name")
    end
    return test_vars
end

"""
    get_SMArtCARE_data(test::String, baseline_df, timedepend_df; extended_output::Bool=false)

Function to preprocess the SMArtCARE data for a specific test. The function returns an `SMATestData` struct with the extracted information 
    on time-dependent and baseline variables, follow-up time points and IDs of all patients for whom the chosen test was conducted.

    From the provided input dataframes, the function first filters the time-dependent dataframe for patients that have the selected test conducted. 
    The dataframe is then subset to the variables of the items of the specific test. 
    The baseline dataframe is subset to the same patients. 
    For each patient, outlier time points are filtered out. 
    An outlier is classified as a time point where the difference to the previous time point is larger than 2 times the 
    interquartile range of all difference between subsequent time points for that patient.
    Additionally, the variance of the sum score of the test is calculated, to allow for potential further subsequent filtering.

# Arguments
    - `test`: name of the motor function test for which the data is collected
    - `baseline_df`: DataFrame containing the baseline variables for all patients
    - `timedepend_df`: DataFrame containing the time-dependent variables for all patients
    - `extended_output`: if `true`, the function also returns the calculated variances of the sumscore for each patient 
        and the time point masks that show which time points where filtered out for each patient. 
"""
function get_SMArtCARE_data(test::String, baseline_df, timedepend_df; extended_output::Bool=false)

    testname="test_$test"
    
    # 1) filter timedepend df 

    # filter for patients that have the selected test conducted 
    timedepend_select_df=timedepend_df[findall(x-> !ismissing(x),timedepend_df[:,testname]),:]
    timedepend_select_df=select(timedepend_select_df,Not(testname))
    # get the variables of the items of the specific test and subset to these 
    test_vars = get_test_variables(test, names(timedepend_select_df))
    # non-item variables that are important 
    other_vars = ["patient_id", "months_since_1st_test"]
    select_vars = vcat(other_vars, test_vars)
    timedepend_select_df=select(timedepend_select_df,select_vars)

    # 2) process baseline variables 

    baseline_vars = names(baseline_df)[findall(x -> !(x ∈ ["cohort", "baseline_date"]), names(baseline_df))]
    select_ids = unique(timedepend_select_df[:,:patient_id])
    baseline_select_df = filter(x -> (x.patient_id ∈ select_ids), baseline_df)
    sort!(baseline_select_df, [:patient_id])
    @assert sort(unique(baseline_select_df[:,:patient_id])) == sort(unique(timedepend_select_df[:,:patient_id]))
    @assert unique(baseline_select_df[:,:patient_id]) == unique(timedepend_select_df[:,:patient_id])
    # recode cohort variable
    for row in 1:nrow(baseline_select_df)
        if baseline_select_df[row,:cohort] == "5"
            baseline_select_df[row, :cohort2] = 9
        end
    end
    baseline_select_df = baseline_select_df[:,baseline_vars]

    # 3) collect timedependent and baseline arrays, skipping outlier time points

    # criterion: calculate IQR of temporal differences of sum score for each patient
    # for each timepoint, remove if the observed difference is outside 2*IQR

    # initialise data containers
    timedepend_xs = []
    tvals = []
    keep_timepoints_masks = []
    baseline_xs = []
    patient_ids = []
    sumscores = []

    for patient_id in select_ids
        # get timedependent variables
        curdf = filter(x -> x.patient_id == patient_id, timedepend_select_df)
        if nrow(unique(curdf)) != nrow(curdf)
            @warn "there were duplicate rows for patient id $(patient_id), skipping these..."
            unique!(curdf)
        end

        # get tvals 
        curtvals = curdf[:,:months_since_1st_test]
        if !(0 in curtvals)
            curtvals.-=minimum(curtvals)
        end
        if length(curtvals) <= 1
            continue
        end
        
        # get sum score and filter time points 
        cursumscore = vec(curdf.rulm_score)
        if var(cursumscore) < 1.0#2.0
            continue 
        end
        curcutoff = 2*iqr(diff(cursumscore))
        curtimepointmask = [true; abs.(diff(cursumscore)) .< curcutoff...]
        if sum(curtimepointmask) <= 1
            continue 
        end
        push!(keep_timepoints_masks, curtimepointmask)
        
        # apply time point mask mask
        curtvals = curtvals[curtimepointmask]
        push!(tvals, vec(curtvals))
        curxs = transpose(Matrix(curdf[curtimepointmask,4:end])) # omitting first, second and third column because they contain ID, timestamp and sum score
        curxs = convert.(Float32, curxs)
        push!(timedepend_xs, curxs)

        # get variance of sumscore 
        cursumscore = cursumscore[curtimepointmask]
        push!(sumscores, cursumscore)

        # get baseline variables 
        curdf_baseline = filter(x -> x.patient_id == patient_id, baseline_select_df)
        curbaselinexs = transpose(Matrix(curdf_baseline[:,2:end])) # again to omit patient_id
        curbaselinexs = vec(curbaselinexs)
        push!(baseline_xs, curbaselinexs)

        # track patient id 
        push!(patient_ids, patient_id)
    end

    # collect into testdata struct 
    testdata = SMATestData(test, 
            convert(Vector{Matrix{Float32}},timedepend_xs),
            convert(Vector{Vector{Float32}},baseline_xs), 
            convert(Vector{Vector{Float32}},tvals), 
            Int.(patient_ids)
    )
    if extended_output 
        return testdata, sumscores, keep_timepoints_masks
    else
        return testdata 
    end
end

logit(p) = log(p) - log(1-p)

"""
    recode_SMArtCARE_data(testdata::SMATestData)

Recodes the time-dependent item values in an `SMATestData` struct to be between 0 and 1. 
    Original item levels are integers between 0 and 2 for all items except item a, which has values between
    0 and 6. 
    Each item is separately mapped to numbers between 0 and 1 and the values are subsequently logit-transformed. 
    A new `SMATestData` struct is returned, where the recoded values are stored in the `xs` field. 

# Arguments
    - `testdata::SMATestData`: the test data to be recoded
    - `recoding_dict`: Dictionary specifying the numbers item levels should be recoded to for all items except a; 
        default is Dict(0 => 0.1, 1 => 0.5, 2 => 0.9) and this is what has been used for all experiments. 
    - `recoding_dict_itema`: Dictionary specifying the numbers item levels should be recoded to for item a; 
        default is Dict(0 => 0.1, 1 => 0.2, 2 => 0.3, 3 => 0.5, 4 => 0.7, 5 => 0.8, 6 => 0.9) and this is what 
        has been used for all experiments.
"""
function recode_SMArtCARE_data(testdata::SMATestData)
    recoding_dict = Dict(0 => 0.1, 1 => 0.5, 2 => 0.9)
    recoding_dict_itema = Dict(0 => 0.1, 1 => 0.2, 2 => 0.3, 3 => 0.5, 4 => 0.7, 5 => 0.8, 6 => 0.9)
    #plot(logit, collect(0:0.001:1))

    item_inds = collect(2:size(testdata.xs[1],1)) # without "itema, which is the first row" 
    recoded_xs = []
    for curxs in testdata.xs 
        @assert issubset(unique(curxs[2:end,:]),  [0, 1, 2])
        cur_recoded_xs = copy(curxs)
        for item_ind in item_inds
            cur_recoded_xs[item_ind,:] = [logit(recoding_dict[item_value]) for item_value in curxs[item_ind,:]]
        end
        # item a 
        cur_recoded_xs[1,:] = [logit(recoding_dict_itema[item_value]) for item_value in curxs[1,:]]
        push!(recoded_xs, cur_recoded_xs)
    end
    recoded_testdata = SMATestData(testdata.test, 
            convert(Vector{Matrix{Float32}},recoded_xs),
            testdata.xs_baseline, 
            testdata.tvals, 
            testdata.ids
    )
    return recoded_testdata
end