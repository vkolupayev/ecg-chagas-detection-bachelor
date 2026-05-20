from calflops import calculate_flops
from src.model.ecg_founder import Net1D
from src.model.hubert import ChagasHuBERT


ecg_founder = Net1D(
    in_channels=12, 
    base_filters=64, #32 64
    ratio=1, 
    filter_list=[64,160,160,400,400,1024,1024],
    m_blocks_list=[2,2,2,3,3,4,4],
    kernel_size=16, 
    stride=2, 
    groups_width=16,
    verbose=False, 
    use_bn=False,
    use_do=False,
    n_classes=1
)


batch_size = 1

input_shape = (batch_size, 12, 5000)
flops, macs, params = calculate_flops(model=ecg_founder, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print(f"ECG Founder FLOPs: {flops}   MACs: {macs}   Params: {params} \n")



hubert_ecg = ChagasHuBERT(
    simple_classifier=True, pretrained=True,
    model_path="pretrained_models/hubert-ecg-small"
)
input_shape = (batch_size, 12000)
flops, macs, params = calculate_flops(model=hubert_ecg, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print(f"HuBERT-ECG  FLOPs: {flops}   MACs: {macs}   Params: {params} \n")