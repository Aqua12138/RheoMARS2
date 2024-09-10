# Target File生成方法
## ShapeMatchingLoss
ShapeMatchingLoss的生成请使用命令
```shell
python fluidlab/run.py --cfg_file configs/exp_gathering.yaml --renderer_type GGUI --record --user_input
```
- 这将会生成一个包含所有帧的x数据以及对应的mat和used
- 这是一个动态的目标

## SDFDensityLoss
SDFDensityLoss的生成请使用命令
```shell
python fluidlab/run.py --cfg_file configs/exp_gathering.yaml --renderer_type GGUI --record_target_grid --user_input
```
- 使用该生成方法会生成出最后一帧的目标的grid mass作为目标grid
- 静态目标
---
# 添加环境
- setup_agent：根据agent config配置agent
- setup_statics：配置静态物体
- setup_bodies：配置流变材料
- setup_boundary：配置流变材料边界
- setup_renderer：配置渲染器
- render
- trainable_policy：仅针对adam算法
---
# 训练方法

- 前置条件，我们提供了两种奖励函数/损失函数方法，请参考Target File生成方法。
## Adam
```shell
python fluidlab/run.py --cfg_file configs/exp_gathering.yaml --renderer_type GGUI --loss_type default --exp_name=adam_gather
```
- Policy需要定义在环境中
## PPO
```shell
python fluidlab/run.py --cfg_file configs/exp_transporting.yaml --renderer_type GGUI --rl ppo --exp_name=ppo_test --perc_type sensor
```
## SHAC
```shell
python fluidlab/run.py --cfg_file configs/shac/default.yaml --renderer_type GGUI --rl shac --exp_name=shac_test --perc_type sensor
```
- 需要定义新的configs文件，包括对环境和shac算法超参数的设置，请参考configs/shac/default.yaml进行设置
- 暂时只考虑了perc_type=sensor的一种配置，请不要切换perc_type为其他
- SHAC环境的horizon要求整除step_nums
---
# 添加Sensor
- Gridsensor3D：参考Mlagent
## 添加方法
- 在/home/zhx/Project/RheoMars/fluidlab/envs/configs目录下的agent.yaml中添加
```yaml
sensors:
  - type: GridSensor3D
    params:
      cell_arc: 2
      lat_angle_north: 90
      lat_angle_south: 90
      lon_angle: 180
      max_distance: 1
      min_distance: 0
      distance_normalization: 1
```
---

# config 配置
## env
- name: 设置环境名称
- seed: 设置环境随机数种子

## loss
- name: Lossl类型
- target_file: 目标分布文件
- weight: 损失权重
- loss_type: adam loss的loss type default/diff

## config
- 训练配置，每种算法不相同，请参考config示例

# 

