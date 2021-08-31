PYTHON3 = python3


# initial condition examples
outputs/ic-small.h5: scripts/run_filter_ic.py data/bz-antispiral-ic-256.h5
	$(PYTHON3) $< --rho 2e-4 --output_file $@

outputs/ic-moderate.h5: scripts/run_filter_ic.py data/bz-antispiral-ic-256.h5
	$(PYTHON3) $< --rho 1e-3 --output_file $@

outputs/ic-large.h5: scripts/run_filter_ic.py data/bz-antispiral-ic-256.h5
	$(PYTHON3) $< --rho 1e-2 --output_file $@

all_outputs_ic: outputs/ic-small.h5 \
	outputs/ic-moderate.h5 \
	outputs/ic-large.h5

plots_ic: scripts/plot_results_ic.py all_outputs_ic
	$(PYTHON3) $< \
		--input_files outputs/ic-small.h5 outputs/ic-moderate.h5 outputs/ic-large.h5 \
		--labels small moderate large \
		--output_dir figures/ic/



# stochastic forcing examples
outputs/stoch-forcing-imex.h5: scripts/run_filter.py
	$(PYTHON3) $< --dt 0.01 --scheme imex --output_file $@

outputs/stoch-forcing-imex-small-dt.h5: scripts/run_filter.py
	$(PYTHON3) $< --dt 0.0001 --scheme imex --output_file $@

outputs/stoch-forcing-cn.h5: scripts/run_filter.py
	$(PYTHON3) $< --dt 0.01 --scheme crank-nicolson --output_file $@

outputs/stoch-forcing-cn-small-dt.h5: scripts/run_filter.py
	$(PYTHON3) $< --dt 0.0001 --scheme crank-nicolson --output_file $@

plots_stoch_forcing_imex: scripts/plot_results.py outputs/stoch-forcing-imex.h5
	$(PYTHON3) $< \
		--input_file outputs/stoch-forcing-imex.h5 \
		--output_dir figures/stoch-forcing-imex/

plots_stoch_forcing_imex_small_dt: scripts/plot_results.py outputs/stoch-forcing-imex-small-dt.h5
	$(PYTHON3) $< \
		--input_file outputs/stoch-forcing-imex-small-dt.h5 \
		--output_dir figures/stoch-forcing-imex-small-dt/
