source("regimes.R")
source("sim_util.R")

method_list <- c("dl", "qlasso", "qrf", "bowl", "sowl")
n_list <- c(100, 200, 400)
scenario_list <- 1 : 5

for (method in method_list) {
  for (n in n_list) {
    for (scenario in scenario_list) {
      # BOWL and SOWL only handles two treatments per stage
      if ((method %in% c("bowl", "sowl")) && (scenario == 5)) next
      
      name <- paste0(method, ".n", n, "s", scenario)
      cat(name, "\n")
      
      assign(name, run.simulation(n = n, scenario = scenario, 
        estimate_regime = get(paste0("estimate.regime.", method)),
        apply_regime = get(paste0("apply.regime.", method))))
      save(list = name, file = paste0(name, ".RData"))
    }
  }
}

