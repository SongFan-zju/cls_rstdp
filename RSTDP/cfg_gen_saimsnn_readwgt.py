import os
import pickle

from Configuration import convConfiguration
from Configuration import readConfiguration
from Node import convNode


leak = 0  # TODO
reset = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.environ.get("PCSS_TB_DIR", script_dir)
output_dir = os.environ.get(
    "SIAMSNN_FLIT_CONFIG_DIR", os.path.join(script_dir, "flit_config")
)
os.makedirs(output_dir, exist_ok=True)

node_list = []
with open(os.path.join(data_dir, "data", "track_weight.pkl"), "rb") as f:
    track_wgt = pickle.load(f)
with open(os.path.join(data_dir, "data", "track_vth.pkl"), "rb") as f:
    track_vth = pickle.load(f)

# conv1 node
node_num = 8
neu_num = 32 * 32
for n in range(0, node_num):
    node_tmp = convNode(fpga=1)
    node_tmp.set_pclass(package_class="write")
    node_tmp.set_node_loc(node_x=0, node_y=0, node_number=n)
    node_tmp.set_neu_par(
        spike_code="LIF", reset=reset, vth=int(track_vth[0][n]), leak=leak
    )
    node_tmp.set_neu_num(neu_num=neu_num)
    node_tmp.set_conv(
        xin=127,
        yin=127,
        xstart=0,
        ystart=0,
        xout=32,
        yout=32,
        zout=n,
        xk=3,
        yk=3,
        pad=0,
        stride=4,
    )  # TODO
    node_tmp.set_wgt_mem(weight_list=track_wgt[0][n])
    node_tmp.set_dst_mem([(1, 1, 1)])
    # node_tmp.set_dst_mem(dst_node_x=2,dst_node_y=1,dst_node_number=1)  # outside
    node_list.append(node_tmp)

# conv2 node
node_num = 16
neu_num = 15 * 15
for n in range(0, node_num):
    node_tmp = convNode(fpga=1)
    node_tmp.set_pclass(package_class="write")
    node_tmp.set_node_loc(node_x=0, node_y=1, node_number=n)
    node_tmp.set_neu_par(
        spike_code="LIF", reset=reset, vth=int(track_vth[1][n]), leak=leak
    )
    node_tmp.set_neu_num(neu_num=neu_num)
    node_tmp.set_conv(
        xin=32,
        yin=32,
        xstart=0,
        ystart=0,
        xout=15,
        yout=15,
        zout=n,
        xk=3,
        yk=3,
        pad=0,
        stride=2,
    )
    node_tmp.set_wgt_mem(weight_list=track_wgt[1][n])
    node_tmp.set_dst_mem([(1, 1, 8)])
    node_list.append(node_tmp)

# conv3 node
node_num = 8
neu_num = 13 * 13
for n in range(0, node_num):
    node_tmp = convNode(fpga=1)
    node_tmp.set_pclass(package_class="write")
    node_tmp.set_node_loc(node_x=1, node_y=1, node_number=n)
    node_tmp.set_neu_par(
        spike_code="LIF", reset=reset, vth=int(track_vth[2][n]), leak=leak
    )
    node_tmp.set_neu_num(neu_num=neu_num)
    node_tmp.set_conv(
        xin=15,
        yin=15,
        xstart=0,
        ystart=0,
        xout=13,
        yout=13,
        zout=n,
        xk=3,
        yk=3,
        pad=0,
        stride=1,
    )
    node_tmp.set_wgt_mem(weight_list=track_wgt[2][n])
    node_tmp.set_dst_mem([(2, 1, 1)])
    node_list.append(node_tmp)

# conf = convConfiguration(node_list=node_list)
# conf.gen_conf(filename=os.path.join(output_dir, "config_saimsnn_r.txt"))

for node_t in node_list:
    node_t.set_pclass(package_class="read")
conf2 = readConfiguration(
    node_list=node_list, read_reg=0, read_wgt=1, read_vm=0, read_dst=0
)
# conf2 = readConfiguration(
#     node_list=node_list, read_reg=0, read_wgt=0, read_vm=1, read_dst=0
# )
conf2.gen_conf(filename=os.path.join(output_dir, "config_read_wgt.txt"))
