out:
	@nvcc -arch=sm_50 main.cu -Wno-deprecated-gpu-targets
	@echo "USAGE: ./a.out -u <use> -h <num_hidden_layers> -n <size_hlayer> -e <epochs> -s <num_samples> -l <learning_rate> -b <batch_size> -o <optimizer> -a <activation_func>"

clean:
	@rm -f a.out
