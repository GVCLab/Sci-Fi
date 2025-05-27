export CUDA_VISIBLE_DEVICES=0
EVAL_DIR=example_input_pairs
MODEL_NAME=CogVideoX-5b-I2V
OUT_DIR=outputs
basename=eval_videos_dir

mkdir -p $OUT_DIR
for example_dir in $(ls -d $EVAL_DIR/*)
do
    example_name=$(basename $example_dir)
    echo $example_name

    out_fn=$OUT_DIR/$example_name'.mp4'
    python Sci_Fi_frame_inbetweening.py \
        --first_image=$example_dir/start.jpg \
        --last_image=$example_dir/end.jpg \
        --EF_Net_model_path='EF_Net/EF_Net.pt' \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --prompt=$example_dir/prompt.txt \
        --out_path=$out_fn \
        --EF_Net_weights=1.0 \
        
done
