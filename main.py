import os 
import argparse
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
from utils import setup_seed, DataManager
from methods import get_learner
def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # General settings
    parser.add_argument('--save_dir', type=str, default="outputs", help='save data')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--g_sigma', type=float, default=0, help='sigma of updata g dp')
    parser.add_argument('--classifer_dp', type=float, default=0, help='dp add to classifer')
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=10, help='num of tasks')
    parser.add_argument('--method', type=str, default="ours_cifar", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=50, help='communication rounds')
    parser.add_argument('--num_users', type=int, default=5, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--local_ep', type=int, default=5, help='local training epochs')
    parser.add_argument('--beta', type=float, default=0.5, help='control the degree of label skew')

    # cul data_mangement
    parser.add_argument('--pre_size', type=int, default=200, help='pre syndata size for per class')
    parser.add_argument('--cur_size', type=int, default=100, help='cur syndata size for per class')
    parser.add_argument('--ul_size', type=int, default=100, help='cur syndata size for ul class')
    parser.add_argument('--ul_local_bs', type=int, default=64, help='local ul batch size')
    parser.add_argument('--ul_local_steps', type=int, default=50, help='local ul steps')

    # hyper
    parser.add_argument('--kd', type=int, default=10, help='kd hyper')
    parser.add_argument('--ul', type=float, default=0.2, help='ul hyper')

    # Diffusion
    parser.add_argument('--g_local_bs', type=int, default=12, help='gen train bs')
    parser.add_argument('--g_local_steps', type=int, default=50, help='gen train steps')
    parser.add_argument('--com_round_gen', type=int, default=10, help='gen train communication rounds')
    parser.add_argument('--n_iter', type=int, default=5, help='500/40 * niter generation samples')
    parser.add_argument('--save_cls_embeds', type=bool, default=True, help='save cls embeds or not')
    #cifar
    parser.add_argument('--config', type=str, default="ldm/ldm_dddr.yaml", help='config of diffusion')
    parser.add_argument('--ldm_ckpt', type=str, default="models/ldm/text2img-large/model.ckpt", help='checkpoint path of latent diffusion model')
  
    args = parser.parse_args()
    return args


args = args_parser()
args.num_class = 100
args.init_cls = int(args.num_class / args.tasks)
args.increment = args.init_cls
args.exp_name = f"beta_{args.beta}_tasks_{args.tasks}_seed_{args.seed}_users_{args.num_users}_g_{args.com_round_gen}_{args.g_local_steps}"
args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, args.exp_name)
args = vars(args)
setup_seed(args["seed"])
data_manager = DataManager(
    args["dataset"],
    False,
    args["seed"],
    args["init_cls"],
    args["increment"],
)
for arg in args:
    print(arg,args[arg])
print('start')
learner = get_learner(args["method"],args)
#task 1
learner.CL_train(data_manager)
learner.eval_task()
learner.log_metrics()
learner.CL_after_task()
#task 2
learner.CL_train(data_manager)
learner.eval_task()
learner.log_metrics()
learner.CL_after_task()
#ul cls_0
learner.UL_train(data_manager,cur_ul_class=0)
learner.eval_task()
learner.log_metrics()
learner.UL_after_task()
#task 3
learner.CL_train(data_manager)
learner.eval_task()
learner.log_metrics()
learner.CL_after_task()
#ul cls_20
learner.UL_train(data_manager,cur_ul_class=20)
learner.eval_task()
learner.log_metrics()
learner.UL_after_task()

print('==============================================================')