outputs/cell-post.h5: scripts/run_cell.py data/rsif-data.xlsx
	python3 $< --data_file data/rsif-data.xlsx --output_file $@

outputs/cell-post-lr-4-modes.h5: scripts/run_cell_lr.py data/rsif-data.xlsx
	python3 $< --n_modes 4 \
		--data_file data/rsif-data.xlsx --output_file $@

outputs/cell-post-lr-8-modes.h5: scripts/run_cell_lr.py data/rsif-data.xlsx
	python3 $< --n_modes 8 \
		--data_file data/rsif-data.xlsx --output_file $@

outputs/cell-post-lr-16-modes.h5: scripts/run_cell_lr.py data/rsif-data.xlsx
	python3 $< --n_modes 16 \
		--data_file data/rsif-data.xlsx --output_file $@

outputs/cell-post-lr-32-modes.h5: scripts/run_cell_lr.py data/rsif-data.xlsx
	python3 $< --n_modes 32 \
		--data_file data/rsif-data.xlsx --output_file $@

outputs/cell-post-lr-64-modes.h5: scripts/run_cell_lr.py data/rsif-data.xlsx
	python3 $< --n_modes 64 \
		--data_file data/rsif-data.xlsx --output_file $@

all_outputs_cell: outputs/cell-post.h5 \
	outputs/cell-post-lr-4-modes.h5 \
	outputs/cell-post-lr-8-modes.h5 \
	outputs/cell-post-lr-16-modes.h5 \
	outputs/cell-post-lr-32-modes.h5 \
	outputs/cell-post-lr-64-modes.h5

plots_cell: all_outputs_cell
	python3 scripts/plot_cell.py \
		--input_file outputs/cell-post-lr.h5 \
		--n_modes 4 8 16 32 64 \
		--input_files_lr outputs/cell-post-lr-4-modes.h5 \
			outputs/cell-post-lr-8-modes.h5 \
			outputs/cell-post-lr-16-modes.h5 \
			outputs/cell-post-lr-32-modes.h5 \
			outputs/cell-post-lr-64-modes.h5 \
		--n_mode_plot 32 --output_dir figures/
