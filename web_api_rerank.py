from time import time
from pprint import pprint
from sentence_transformers import CrossEncoder
from flask import Flask, jsonify, request
from flask_ipfilter import IPFilter, Whitelist


'''
Flask Web API
'''
# Build Flask app
app = Flask(__name__)

# Set whitelist for IP filter
ip_filter = IPFilter(app, ruleset=Whitelist())
ip_filter.ruleset.permit("127.0.0.1")

# Re-ranker
cross_encoder = CrossEncoder(
    'BAAI/bge-reranker-v2-m3', # You can change to other models here
    device='cpu', # 'cuda:0'
    trust_remote_code=True
)

# Re-rank text content
@app.route("/rerank", methods = ["POST"])
def rerank():
    # Get request data
    q = request.json['q']
    li_sentences = request.json['li_sentences']
    li_urls = request.json['li_urls']

    # Re-rank
    t1 = time()
    print('Reranking...')
    ranks = cross_encoder.rank(q, li_sentences, return_documents=True)
    for index, obj in enumerate(ranks):
        ranks[index]['score'] = float(obj['score'])
        ranks[index]['url'] = li_urls[obj['corpus_id']]
    pprint(ranks)
    print(f"Reranking took: {time() - t1:.2f} seconds")

    # Return results
    return jsonify({"ranks": ranks})

# Main program area
if __name__ == '__main__':
    app.debug = False
    app.json.ensure_ascii = False
    app.run(
        host='127.0.0.1', # 0.0.0.0 
        port=5004,
        threaded=True
    )

