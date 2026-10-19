"""Show a jupyterquiz quiz one question at a time, with Previous/Next buttons.

Usage in a (remove-input) code cell:

    import sys; sys.path.append("..")
    from quizpager import display_quiz_paged
    display_quiz_paged("quiz/error_types.json")
"""
import uuid

from IPython.display import HTML, Javascript, display
from jupyterquiz import display_quiz

_PAGER_JS = r"""
(function () {
  // The quiz container is the first jupyterquiz div after our marker
  function findContainer() {
    const marker = document.getElementById("__MARKER__");
    if (!marker) return null;
    for (const el of document.querySelectorAll("div[data-numquestions]")) {
      if (marker.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING) return el;
    }
    return null;
  }

  function setup(tries) {
    const container = findContainer();
    const questions = container ? container.querySelectorAll(":scope > .Quiz") : [];
    if (!questions.length) {
      if (tries < 100) setTimeout(() => setup(tries + 1), 100);
      return;
    }
    if (container.dataset.paged) return;
    container.dataset.paged = "true";

    const nav = document.createElement("div");
    nav.className = "JQPager";
    nav.innerHTML =
      '<button type="button" class="JQPagerBtn">&#9664; Previous</button>' +
      '<span class="JQPagerCount"></span>' +
      '<button type="button" class="JQPagerBtn">Next &#9654;</button>';
    container.appendChild(nav);

    const [prev, count, next] = nav.children;
    let current = 0;

    function show(i) {
      current = i;
      questions.forEach((q, j) => { q.style.display = j === i ? "" : "none"; });
      count.textContent = `Question ${i + 1} of ${questions.length}`;
      prev.disabled = i === 0;
      next.disabled = i === questions.length - 1;
    }

    prev.addEventListener("click", () => show(current - 1));
    next.addEventListener("click", () => show(current + 1));
    show(0);
  }

  setup(0);
})();
"""

_PAGER_CSS = """
<style>
.JQPager {
    display: flex;
    align-items: center;
    justify-content: space-between;
    max-width: 600px;
    margin: 10px auto 15px auto;
    font-size: 14px;
}
.JQPagerBtn {
    background-color: var(--jq-multiple-choice-bg);
    color: var(--jq-text-color);
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    cursor: pointer;
}
.JQPagerBtn:disabled {
    opacity: 0.35;
    cursor: default;
}
</style>
"""


def display_quiz_paged(ref, **kwargs):
    """Like jupyterquiz.display_quiz, but shows one question at a time."""
    marker = "jqpager" + uuid.uuid4().hex[:8]
    display(HTML(f'<div id="{marker}"></div>' + _PAGER_CSS))
    display_quiz(ref, **kwargs)
    display(Javascript(_PAGER_JS.replace("__MARKER__", marker)))
