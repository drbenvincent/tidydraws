# Copyright (c) 2026 Benjamin Vincent
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Crawl the published documentation with bounded retries and local fragments."""

from collections import deque
from html.parser import HTMLParser
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse, unquote
from urllib.request import Request, urlopen

BASE = "https://drbenvincent.github.io/tidydraws/"
# Great Docs 0.15 emits these four cross-reference fragments without target IDs.
# Still check each page; exempt only these known generated fragments.
KNOWN_MISSING_FRAGMENTS = {
    urljoin(BASE, f"reference/{name}.html#tidydraws.{name}")
    for name in (
        "parameter_draws",
        "prediction_draws",
        "compare_draws",
        "point_interval",
    )
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "tidydraws-link-check"}), timeout=20
            ) as response:
                content = response.read()
                return (
                    content.decode("utf-8")
                    if "text/html" in response.headers.get("Content-Type", "")
                    else None
                )
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main():
    pending = deque([
        BASE,
        *(urljoin(BASE, f) for f in ("llms.txt", "llms-full.txt", "skill.md")),
    ])
    cache = {}
    errors = []
    while pending:
        link = pending.popleft()
        url, fragment = urldefrag(link)
        if urlparse(url).scheme not in {"http", "https"}:
            continue
        if url not in cache:
            try:
                body = fetch(url)
                parser = Links()
                if body is not None:
                    parser.feed(body)
                cache[url] = parser
                if url.startswith(BASE):
                    pending.extend(urljoin(url, href) for href in parser.links)
            except (HTTPError, URLError, TimeoutError) as error:
                cache[url] = None
                errors.append(f"{url}: {error}")
        parser = cache[url]
        if (
            fragment
            and parser
            and url.startswith(BASE)
            and link not in KNOWN_MISSING_FRAGMENTS
        ):
            if unquote(fragment) not in parser.ids:
                errors.append(f"Missing fragment: {link}")
    if errors:
        raise RuntimeError("\n".join(sorted(set(errors))))
    print(f"Checked {len(cache)} published URLs")


if __name__ == "__main__":
    main()
