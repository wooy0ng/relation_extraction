

def entity_marker_punct(subj: str, obj: str, sentence: str) -> str:
    e1_start_idx = sentence.find(subj)
    e2_start_idx = sentence.find(obj)
    
    if e1_start_idx > e2_start_idx:
        e1_start_idx, e2_start_idx = e2_start_idx, e1_start_idx
        subj, obj = obj, subj
    sentence = sentence[:e1_start_idx] + f'@{subj}@' + \
                sentence[e1_start_idx+len(subj):e2_start_idx] + f'#{obj}#' + \
                sentence[e2_start_idx+len(obj):]
    return sentence