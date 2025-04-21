from nuscenes.nuscenes import NuScenes

def get_all_sample_tokens(NuScenes, scene_token):
    """
    遍历给定场景中的所有样本token。

    参数:
    level5data (LyftDataset): LyftDataset对象。
    scene_token (str): 场景token。

    返回:
    list: 场景中的所有样本token列表。
    """
    # 获取场景记录
    scene = NuScenes.get("scene", scene_token)
    
    # 初始化样本token列表
    sample_tokens = []
    
    # 获取第一个样本token
    sample_token = scene["first_sample_token"]
    
    # 循环遍历所有样本token
    while sample_token:
        sample_tokens.append(sample_token)
        sample_record = NuScenes.get("sample", sample_token)
        sample_token = sample_record["next"]
    
    return sample_tokens