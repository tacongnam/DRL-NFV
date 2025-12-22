from data_info.read_data import Read_data
import sys
sys.path.append('..')


if __name__ == "__main__":
    print(sys.path)
    PATH = "data_1_9/cogent_centers_easy_s1.json"
    reader = Read_data(PATH)
    topology = reader.get_E(len(reader.get_V()))
    for i in range(topology.num_dcs):
        for j in range(topology.num_dcs):
            print(f"Delay from DC {i} to DC {j}: {topology.get_propagation_delay(i, j)} ms")