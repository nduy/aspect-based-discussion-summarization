from functions import *
from decoration import *
from datetime import datetime as dt
from datetime import timedelta as td
from config import *
from community_detect import *
# ------ Time recording
import time

start_time = time.time()
if __name__ == "__main__":
    comments, comment_js, comment_des = read_comment_file("data/comments_article23.txt", read_as_threads=False)
    title, article = read_article_file("data/article23.txt")

    dataset = {'title': title,
               'article': article,
               'comments': comments}

    clusters = {}
    cluster_number = 0
    # now run text titling
    counters = []
    nodes = []
    json_g = {
        'nodes':[],
        'edges':[]
    }
    for segment in texttiling_tokenize(article):  # Run texttiling, then go to each segment
        # Get 10 top frequent NN, NNS
        sent_tokenize_list = sent_tokenize(segment)
        c = Counter()
        for sentence in sent_tokenize_list:
            b = TextBlob(sentence)
            items = [w for w, t in b.tags if t == u'NN' or t == u'NNS'] + list(b.noun_phrases)
            c.update(items)
        counters.append(c)
        top_10 = c.most_common(10)
        cluster_id = ' - '.join([w for w, c in top_10])
        top_ids = set([w + '~'+ gen_mcs() for w,c in top_10 if re.match('^([a-z_\s,-]+).',w)])
        # if not top_ids: continue
        # print top_ids
        for item in top_ids:
            # print item
            if re.match('^([a-z_\s,-]+)~',item):
                json_g['nodes'].append({
                    "cid": cluster_id ,
                    "color": "#0080ff",
                    "group": "article",
                    "id": item,
                    "label": re.match('^([a-z_\s,-]+)~',item).group(1),
                    "title": segment,
                    "value": 1,
                    "sen_score": 0.0
                })
        clusters[cluster_id] = top_ids

    for comment in comments:
        # print comment_id
        sent_tokenize_list = sent_tokenize(comment['content'])
        c = Counter()
        for sentence in sent_tokenize_list:
            b = TextBlob(sentence)
            items = [w for w, t in b.tags if t == u'NN' or t == u'NNS'] + list(b.noun_phrases)
            c.update(items)
        keywords = [w for w, c in c.most_common(10)]
        edges = set()
        for cluster_id in clusters:
            for member in clusters[cluster_id]:
                #print member, keywords
                if re.match('^([a-z_\s,-]+)~',member):
                    item = re.match('^([a-z_\s,-]+)~',member).group(1)
                    if item in keywords:
                        # print member,comment_id
                        js = {
                                'from': member,
                                'id': 'n2cmn~'+member+'~'+comment['comment_id'] + '~' + gen_mcs_only(),
                                'label': '',
                                'title': '',
                                'to': 'comment~'+comment['comment_id'],
                                'value': 1,
                                'arrows': { 'to': { 'enabled' : False}, 'from' : {'enabled': False}},
                                'dashes': True,
                                'smooth': {'type': 'cubicBezier','forceDirection': 'none','roundness': 0.5},
                                'hoverWidth': 1.5,
                                'labelHighlightBold': True,
                                'physics': False,
                                'width': 0.1,
                                'hoverWidth': 0.2
                        }
                        if js not in json_g['edges']:
                            json_g['edges'].append(js)
    # json_g['edges'].extend()
    # Add build options
    json_g['options'] = {
        'timestamp': dt.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        'build_option': build_options,
        'pruning_option': prune_options,
        'unification_option': uni_options,
        'community_detection_option': community_detect_options
    }
    json_g['summary'] = {
        'n_comments': len(comments)
    }

    # Add edges from nodes to comments
     # print(comment_mean_sentiment)
    # add comments
    json_g['comments'] = comment_js  # add comment descriptions
    json_g['nodes'].extend(comment_des)

    with open('result_texttiling.json', 'w') as outfile:
        json.dump(json_g, outfile, sort_keys=True, indent=4, separators=(',', ': '))


