import nltk


def find_all_substring(og_text: str, own_entity: str):
    ret = []
    start_pos = og_text.find(own_entity)
    if start_pos == -1:
        return ret
    else:
        return ret + [[start_pos, start_pos+len(own_entity)]]


def my_sentence_tokenizer(original_text, entity_poses):
    """
    Input
        original_text: str, text_title + '\t' + text_abstract
        entity_poses: a list of elements [start_pos, end_pos] such that original_text[start_pos:end_pos] is a named entity, so that they shouldn't be split.
    Output
        a list of positions [sent_start, sent_end] such that each original_text[sent_start:sent_end] is a sentence.

    """
    text_title, text_abstract = original_text.split('\t')
    extra_abbreviations = ['e.g', 'i.e', 'i.m', 'a.u', 'p.o', 'i.v', 'i.p', 'vivo', 'p.o', 'i.p', 'Vmax', 'i.c.v', ')(', 'E.C', 'sp']
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    own_entity_list = [
        "no. 4, 1,3,5(10)-estratriene-17alpha-ethynyl-3", 
        "no. 6, 17alpha-ethynyl-androstene-diol", 
        "no. 8, 3beta, 17beta-dihydroxy-androst-5-ene-16-one",
        "no. 10, 3beta-methylcarbonate-androst-5-ene-7,17-dione",
        "4-DAMP. 1.5 +/- 0.4 nM",
        "(Mecp2(tm)(1)(.)(1)(Jae))",
        "(Bioorg. Med. Chem., 2010, 18, 1761-1772., J. Med. Chem., 2011, 54, 2823-2838.)"
    ]

    sentence_poses = sentence_tokenizer.span_tokenize(text_title)
    end_title_pos = len(text_title) - 1
    sentence_poses = list(sentence_poses)
    abstract_sentence_poses = sentence_tokenizer.span_tokenize(text_abstract)
    for start_pos, end_pos in abstract_sentence_poses:
        sentence_poses.append((start_pos+end_title_pos+2, end_pos+end_title_pos+2))

    og_text = '\t'.join([text_title, text_abstract])
    for own_entity in own_entity_list:
        entity_poses += find_all_substring(og_text, own_entity)
    is_sentence_break = [True] * (len(sentence_poses) - 1)
    new_sentence_poses = []

    for entity_start, entity_end in entity_poses:
        entity_start = int(entity_start)
        entity_end = int(entity_end)
        for sentence_idx in range(len(sentence_poses)-1):
            sentence_end = sentence_poses[sentence_idx][1]
            if entity_start <= sentence_end and sentence_end < entity_end:
                is_sentence_break[sentence_idx] = False
    
    sentence_startends = []
    for sent_idx, (sent_start, sent_end) in enumerate(sentence_poses):
        sentence_startends.append(sent_start)
        sentence_startends.append(sent_end)

    is_valid_startends = [True]
    for is_sent_break in is_sentence_break:
        is_valid_startends.append(is_sent_break)
        is_valid_startends.append(is_sent_break)
    is_valid_startends.append(True)
    sentence_startends = [sentence_startends[idx] for idx in range(len(sentence_startends)) if is_valid_startends[idx] == True]
    new_sentence_poses = []
    for idx in range(len(sentence_startends)//2):
        new_sentence_poses.append((sentence_startends[2*idx], sentence_startends[2*idx+1]))

    return new_sentence_poses
