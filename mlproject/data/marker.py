

def entity_marker_punct(subj: str, obj: str, sentence: str) -> str:
    e1_start_idx = sentence.find(subj)
    e2_start_idx = sentence.find(obj)
    
    subj = f'@{subj}@'
    obj = f'#{obj}#'
    if e1_start_idx > e2_start_idx:
        e1_start_idx, e2_start_idx = e2_start_idx, e1_start_idx
        subj, obj = obj, subj
        
    '''
    subject : 비틀즈
    object : 조지 해리슨
        before : '〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.'
        after : '〈Something〉는 #조지 해리슨#이 쓰고 @비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.'
    '''
    sentence = sentence[:e1_start_idx] + subj + \
                sentence[e1_start_idx+len(subj)-2:e2_start_idx] + obj + \
                sentence[e2_start_idx+len(obj)-2:]
    return sentence