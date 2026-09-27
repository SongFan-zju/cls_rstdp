import os
import pickle

from Configuration import readConfiguration
from Configuration import uniConfiguration
from Node import uniNode


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.environ.get("PCSS_TB_DIR", script_dir)
output_dir = os.environ.get(
    "RSTDP_FLIT_CONFIG_DIR", os.path.join(script_dir, "flit_config")
)
os.makedirs(output_dir, exist_ok=True)

node_list = []

neu_num = 5
reset = 1
leak = 0
vth = 127

with open(os.path.join(data_dir, "data", "axon.pkl"), "rb") as f:
    axon_mem = pickle.load(f)
with open(os.path.join(data_dir, "data", "weight_init.pkl"), "rb") as f:
    weights_mem = pickle.load(f)

node_tmp = uniNode(reset_ram=1)
node_tmp.set_pclass(package_class="write")
node_tmp.set_node_loc(node_x=1, node_y=1, node_number=35)
node_tmp.set_neu_par(reset=reset, vth=vth, leak=leak, ref_period=0)
node_tmp.set_neu_num(neu_num=neu_num)
# node_tmp.set_conv(xin=127,yin=127,xstart=0,ystart=0,xout=32,yout=32,zout=n,xk=3,yk=3,pad=0,stride=4) # TODO
node_tmp.set_uni(axon_mem=axon_mem)
node_tmp.set_wgt_mem(weight_list=weights_mem)
node_tmp.set_dst_mem([(2, 1, 1)])
node_tmp.set_stdp(lrn_period=8, a_minus=0, a_plus=63, t_const=1)
# node_tmp.set_dst_mem(dst_node_x=2,dst_node_y=1,dst_node_number=1)  # outside
node_list.append(node_tmp)

# conf = uniConfiguration(node_list=node_list)
# conf.gen_conf(filename=os.path.join(output_dir, "config_uni_rstdp.txt"))

for node_t in node_list:
    node_t.set_pclass(package_class="read")
conf2 = readConfiguration(
    node_list=node_list, read_reg=0, read_wgt=1, read_vm=0, read_dst=0
)
conf2.gen_conf(filename=os.path.join(output_dir, "config_read_uni_rstdp.txt"))
