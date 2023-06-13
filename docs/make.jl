using Documenter
using LatentDynamics

makedocs(
    sitename = "LatentDynamics.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md",
        "Data processing" => "data.md",
        "Dynamic model" => "dynamics.md",
        "VAE model" => "vae.md",
        "Optimization" => "optimization.md",
        "Evaluation" => "evaluation.md",
        "Plotting" => "plotting.md"
    ]
)

deploydocs(
    repo = "github.com/maren-ha/LatentDynamics.jl.git",
    devbranch = "main"
)