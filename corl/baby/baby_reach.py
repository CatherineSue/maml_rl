from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from maml_examples.point_env_randgoal import PointEnvRandGoal
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv


N_TASKS = 50
TASKNAME = 'reach'

learning_rates = [1e-2]

fast_learning_rates = [0.5]
baselines = ['linear']
fast_batch_size = 20  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
meta_batch_size = 40  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 100
num_grad_updates = 1
meta_step_size = 0.01

use_maml = True

for fast_learning_rate in fast_learning_rates:
    for learning_rate in learning_rates:
        for bas in baselines:
            stub(globals())

            goal_low = np.array((-0.1, 0.8, 0.05))
            goal_high = np.array((0.1, 0.9, 0.3))

            goals = np.random.uniform(low=goal_low, high=goal_high, size=(N_TASKS, len(goal_low))).tolist()
            print(goals)

            tasks =[
                {'goal': np.array(g), 'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'reach'}
                for i, g in enumerate(goals)
            ]

            env = TfEnv(normalize(SawyerReachPushPickPlace6DOFEnv(
                random_init=False,
                multitask=False,
                obs_type='plain',
                if_render=False,
                tasks=tasks,
            )))
            policy = MAMLGaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                grad_step_size=fast_learning_rate,
                hidden_nonlinearity=tf.nn.relu,
                hidden_sizes=(100,100),
            )
            if bas == 'zero':
                baseline = ZeroBaseline(env_spec=env.spec)
            elif 'linear' in bas:
                baseline = LinearFeatureBaseline(env_spec=env.spec)
            else:
                baseline = GaussianMLPBaseline(env_spec=env.spec)
            algo = MAMLTRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=fast_batch_size, # number of trajs for grad update
                max_path_length=max_path_length,
                meta_batch_size=meta_batch_size,
                num_grad_updates=num_grad_updates,
                n_itr=100,
                use_maml=use_maml,
                step_size=meta_step_size,
                plot=False,
            )
            run_experiment_lite(
                algo.train(),
                n_parallel=1,
                snapshot_mode="last",
                python_command='python3',
                seed=1,
                exp_prefix='maml_rl_trpo_baby_{}'.format(TASKNAME),
                exp_name='trpomaml'+str(int(use_maml))+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + 'metalr_' + str(meta_step_size) +'_step1'+str(num_grad_updates),
                plot=False,
            )
