outputs/cell-post.h5: scripts/run_cell.py data/rsif-data.xlsx
	python3 $< --data_file data/rsif-data.xlsx --output_file $@

outputs/cell-post-lr-%-modes.h5: scripts/run_cell_lr.py data/rsif-data.xlsx
	python3 $< --n_modes $* --n_modes_prior 64 \
		--data_file data/rsif-data.xlsx --output_file $@

outputs/cell-post-lr-%-modes-prior.h5: scripts/run_cell_lr.py data/rsif-data.xlsx
	python3 $< --n_modes 32 --n_modes_prior $* \
		--data_file data/rsif-data.xlsx --output_file $@

all_outputs_cell: outputs/cell-post.h5 \
	outputs/cell-post-lr-4-modes.h5 \
	outputs/cell-post-lr-8-modes.h5 \
	outputs/cell-post-lr-16-modes.h5 \
	outputs/cell-post-lr-32-modes.h5 \
	outputs/cell-post-lr-48-modes.h5 \
	outputs/cell-post-lr-64-modes.h5 \
	outputs/cell-post-lr-4-modes-prior.h5 \
	outputs/cell-post-lr-8-modes-prior.h5 \
	outputs/cell-post-lr-16-modes-prior.h5 \
	outputs/cell-post-lr-32-modes-prior.h5 \
	outputs/cell-post-lr-48-modes-prior.h5 \
	outputs/cell-post-lr-64-modes-prior.h5

clean_all_outputs_cell:
	rm outputs/cell-post.h5 \
		outputs/cell-post-lr-4-modes.h5 \
		outputs/cell-post-lr-8-modes.h5 \
		outputs/cell-post-lr-16-modes.h5 \
		outputs/cell-post-lr-32-modes.h5 \
		outputs/cell-post-lr-48-modes.h5 \
		outputs/cell-post-lr-64-modes.h5 \
		outputs/cell-post-lr-4-modes-prior.h5 \
		outputs/cell-post-lr-8-modes-prior.h5 \
		outputs/cell-post-lr-16-modes-prior.h5 \
		outputs/cell-post-lr-32-modes-prior.h5 \
		outputs/cell-post-lr-48-modes-prior.h5 \
		outputs/cell-post-lr-64-modes-prior.h5

figures/cell-lr-modes-sigma.png: scripts/run_cell_lr_parameterised.py
	python3 $< --data_file data/rsif-data.xlsx --output_file $@

plots_cell: scripts/plot_cell.py all_outputs_cell
	python3 $< \
		--input_file outputs/cell-post.h5 \
		--n_modes 4 8 16 32 48 64 \
		--input_files_lr outputs/cell-post-lr-4-modes.h5 \
			outputs/cell-post-lr-8-modes.h5 \
			outputs/cell-post-lr-16-modes.h5 \
			outputs/cell-post-lr-32-modes.h5 \
			outputs/cell-post-lr-48-modes.h5 \
			outputs/cell-post-lr-64-modes.h5 \
		--n_mode_plot 64 \
		--output_dir figures/

figures/cell-lr-modes-errors.png: scripts/plot_cell_errors_modes.py all_outputs_cell
	python3 $< \
		--input_file outputs/cell-post.h5 \
		--n_modes 4 8 16 32 48 64 \
		--input_files_lr outputs/cell-post-lr-4-modes.h5 \
			outputs/cell-post-lr-8-modes.h5 \
			outputs/cell-post-lr-16-modes.h5 \
			outputs/cell-post-lr-32-modes.h5 \
			outputs/cell-post-lr-48-modes.h5 \
			outputs/cell-post-lr-64-modes.h5 \
		--output_file $@

figures/cell-lr-modes-errors-prior.png: scripts/plot_cell_errors_modes.py all_outputs_cell
	python3 $< \
		--input_file outputs/cell-post.h5 \
		--n_modes 4 8 16 32 48 64 \
		--prior_modes \
		--input_files_lr outputs/cell-post-lr-4-modes-prior.h5 \
			outputs/cell-post-lr-8-modes-prior.h5 \
			outputs/cell-post-lr-16-modes-prior.h5 \
			outputs/cell-post-lr-32-modes-prior.h5 \
			outputs/cell-post-lr-48-modes-prior.h5 \
			outputs/cell-post-lr-64-modes-prior.h5 \
		--output_file $@
