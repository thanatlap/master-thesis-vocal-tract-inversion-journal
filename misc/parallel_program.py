import numpy as np
import multiprocessing as mp

def worker(input_x, input_y, output, output_2):
    out = ['f'+str(feature) for i, feature in enumerate(input_x)]
    out2 = ['f'+str(feature) for i, feature in enumerate(input_y)]
    output.put(out)
    output_2.put(out2)

def main():
	input_x = list(range(100))
	input_y = list(range(100))
	njob = 6

	output = mp.Queue()
	output_2 = mp.Queue()

	processes = []
	for i in range(njob):
	    start = i*int(len(input_x)/njob)
	    end = (i+1)*int(len(input_x)/njob) if i != njob-1 else len(input_x)
	    # Setup a list of processes that we want to run
	    processes.append(mp.Process(target=worker, args=(input_x[start:end], input_y[start:end], output, output_2)))

	# Run processes
	for p in processes:
	    p.start()

	# Exit the completed processes
	for p in processes:
	    p.join()

	out = []
	out2 = []
	for p in processes:
		out.extend(output.get())
		out2.extend(output_2.get())
	print(out)
	print(out2)

if __name__ == '__main__':
	main()