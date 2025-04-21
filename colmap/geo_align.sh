path=$1

mkdir ${path}/aligned_model
# mv ${path}/sparse_model ${path}/sparse

colmap model_aligner \
    --input_path ${path}/sparse/0 \
    --output_path ${path}/aligned_model/ \
    --transform_path ${path}/transform.txt \
    --ref_images_path ${path}/geo_registration/geo_registration.txt \
    --ref_is_gps 0 \
    --alignment_type custom \
    --alignment_max_error 3 

