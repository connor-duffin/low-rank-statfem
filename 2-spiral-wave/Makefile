# spiral gendata example
outputs/spiral-post-gendata.h5: scripts/run_spiral_gendata.py \
	data/bz-spiral-data.h5 data/bz-spiral-ic.h5
	python3 $< \
		--data_file data/bz-spiral-data.h5 --output_file $@

plots_gendata: scripts/plot_spiral_gendata.py
	python3 $< \
		--data_file data/bz-spiral-data.h5 \
		--input_file outputs/spiral-post-gendata.h5 \
		--output_dir figures/spiral-gendata/


# spiral divergence example
outputs/spiral-divergence-crank-nicolson.h5:
	python3 scripts/run_spiral_divergence.py \
		--scheme crank-nicolson --output_file $@

outputs/spiral-divergence-imex.h5:
	python3 scripts/run_spiral_divergence.py \
		--scheme imex --output_file $@

plots_divergence: scripts/plot_spiral_divergence.py \
	outputs/spiral-divergence-crank-nicolson.h5 \
	outputs/spiral-divergence-imex.h5
	python3 $< \
		--input_file_cn outputs/spiral-divergence-crank-nicolson.h5 \
		--input_file_imex outputs/spiral-divergence-imex.h5 \
		--output_dir figures/spiral-divergence/

