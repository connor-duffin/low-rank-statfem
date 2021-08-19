outputs/cell-post.h5: scripts/run_cell.py
	python3 $< --output_file $@

outputs/cell-post-lr-4-modes.h5: scripts/run_cell_lr.py data/rsif-formatted.xlsx
	python3 $< --n_modes 4 --output_file $@

outputs/cell-post-lr-8-modes.h5: scripts/run_cell_lr.py data/rsif-formatted.xlsx
	python3 $< --n_modes 8 --output_file $@

outputs/cell-post-lr-16-modes.h5: scripts/run_cell_lr.py data/rsif-formatted.xlsx
	python3 $< --n_modes 16 --output_file $@

outputs/cell-post-lr-32-modes.h5: scripts/run_cell_lr.py data/rsif-formatted.xlsx
	python3 $< --n_modes 32 --output_file $@

outputs/cell-post-lr-64-modes.h5: scripts/run_cell_lr.py data/rsif-formatted.xlsx
	python3 $< --n_modes 64 --output_file $@

plots_cell:
	python3 scripts/plot_cell.py \
		--input_file outputs/cell-post-lr.h5 \
		--n_modes 4 8 16 32 64 \
		--input_files_lr outputs/cell-post-lr-4-modes.h5 \
			outputs/cell-post-lr-8-modes.h5 \
			outputs/cell-post-lr-16-modes.h5 \
			outputs/cell-post-lr-32-modes.h5 \
			outputs/cell-post-lr-64-modes.h5 \
		--n_mode_plot 32 --output_dir figures/
