from gnews import GNews

gn = GNews(language='vi', country='VN', max_results=10)
results = gn.get_news('Vinamilk cổ phiếu VNM')

print(f'Found: {len(results)} articles')
for r in results[:3]:
    print(f"- {r['title']}")
    print(f"  {r['published date']}")
    print()
