from ddgs import DDGS


def search_web(query, max_results=3):
    print(f"  Searching web for: '{query}'")

    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })

    print(f"  Found {len(results)} web results.")
    return results


def format_web_results(results):
    if not results:
        return "No web results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"[{i}] {r['title']}\n{r['snippet']}\nSource: {r['url']}")

    return "\n\n".join(formatted)
