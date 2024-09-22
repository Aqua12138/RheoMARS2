#!/bin/bash

# ��ȡ����Ĳ���
CFG_FILE=""
RENDERER_TYPE=""
RL=""
EXP_NAME=""
PERC_TYPE=""
PRE_TRAIN_MODEL=""
INITIAL_MODEL_FLAG=false
START_NUM=1
END_NUM=5

# ���������в���
while [[ $# -gt 0 ]]; do
  case $1 in
    --cfg_file)
      CFG_FILE="$2"
      shift
      shift
      ;;
    --renderer_type)
      RENDERER_TYPE="$2"
      shift
      shift
      ;;
    --rl)
      RL="$2"
      shift
      shift
      ;;
    --exp_name)
      EXP_NAME="$2"
      shift
      shift
      ;;
    --perc_type)
      PERC_TYPE="$2"
      shift
      shift
      ;;
    --pre_train_model)
      PRE_TRAIN_MODEL="$2"
      shift
      shift
      ;;
    --intial_model)
      INITIAL_MODEL_FLAG=true
      shift
      ;;
    --start_num)
      START_NUM="$2"
      shift
      shift
      ;;
    --end_num)
      END_NUM="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# ����Ҫ�����Ƿ����
if [[ -z $CFG_FILE || -z $RENDERER_TYPE || -z $RL || -z $EXP_NAME || -z $PERC_TYPE || -z $PRE_TRAIN_MODEL ]]; then
  echo "ȱ�ٱ�Ҫ�Ĳ������������롣"
  exit 1
fi

# ��� start_num �� end_num �Ƿ����
if [[ $START_NUM -gt $END_NUM ]]; then
  echo "����: --start_num ���ܴ��� --end_num"
  exit 1
fi

# ѭ������ָ������
for ((i=START_NUM; i<=END_NUM; i++)); do
  # ��������
  CMD="python run.py --cfg_file $CFG_FILE --renderer_type $RENDERER_TYPE --rl $RL --exp_name ${EXP_NAME}_$i --perc_type $PERC_TYPE --pre_train_model $PRE_TRAIN_MODEL"
  
  # ��������� initial_model ��־������ӵ�������
  if [[ $INITIAL_MODEL_FLAG == true ]]; then
    CMD="$CMD --intial_model"
  fi

  # ��������
  echo "��������: $CMD"
  $CMD
done
