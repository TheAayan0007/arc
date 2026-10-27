"""
web_agent.py - Selenium browsing for the agent.
"""
import re
import json
import time
import socket
import threading
from datetime import datetime
from urllib.parse import urlparse, quote_plus, unquote

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_OK = True
except ImportError:
    SELENIUM_OK = False

PHONE_RE = re.compile(r"(?<![\w/])(\+?\(?\d[\d ().\-\u00a0]{6,18}\d)(?![\w])")
DATE_LIKE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{1,2}[-./]\d{1,2}[-./]\d{2,4}$")


def extract_phones(text):
    found = {}
    for m in PHONE_RE.finditer(text or ""):
        raw = m.group(1).strip()
        digits = re.sub(r"\D", "", raw)
        if not 8 <= len(digits) <= 15:
            continue
        if DATE_LIKE.match(raw) or len(set(digits)) <= 2:
            continue
        found.setdefault(digits, raw)
    return list(found.values())


def normalize_url(text):
    t = text.strip()
    if re.match(r"^[a-z][a-z0-9+.-]*://", t, re.I):
        return t
    if " " in t or "." not in t:
        return "https://www.google.com/search?q=" + quote_plus(t)
    return "https://" + t


SEARCH_BOX = [
    "input[type='search']", "[role='searchbox']", "input[name='q']", "textarea[name='q']",
    "input[name='query']", "input[name='search']", "input[name='s']", "input[name='keyword']",
    "input[aria-label*='search' i]", "input[placeholder*='search' i]",
    "input[id*='search' i]", "input[class*='search' i]", "[role='search'] input",
]
SEARCH_TRIGGER = [
    "button[aria-label*='search' i]", "a[aria-label*='search' i]", "button[title*='search' i]",
    "button[class*='search' i]", "a[class*='search' i]", "[role='button'][class*='search' i]",
    "[id*='search' i]:not(input):not(form)", "[title*='search' i]",
]

JS_HIGHLIGHT = r"""
const phrase = arguments[0];
document.querySelectorAll('mark.__ab_hl').forEach(m => {
  const p = m.parentNode; p.replaceChild(document.createTextNode(m.textContent), m); p.normalize();
});
const esc = phrase.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+');
const re = new RegExp(esc, 'gi');
const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
  acceptNode: n => {
    const p = n.parentElement; if (!p) return NodeFilter.FILTER_REJECT;
    return ['SCRIPT','STYLE','NOSCRIPT','TEXTAREA'].includes(p.tagName) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
  }});
const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
let count = 0;
nodes.forEach(n => {
  const txt = n.nodeValue; re.lastIndex = 0; if (!re.test(txt)) return; re.lastIndex = 0;
  const frag = document.createDocumentFragment(); let last = 0, m;
  while ((m = re.exec(txt))) {
    if (m[0].length === 0) { re.lastIndex++; continue; }
    frag.appendChild(document.createTextNode(txt.slice(last, m.index)));
    const mk = document.createElement('mark'); mk.className = '__ab_hl';
    mk.style.cssText = 'background:#ffeb3b;color:#000;outline:2px solid #ff9800;';
    mk.textContent = m[0]; frag.appendChild(mk); last = m.index + m[0].length; count++;
  }
  frag.appendChild(document.createTextNode(txt.slice(last)));
  n.parentNode.replaceChild(frag, n);
});
const first = document.querySelector('mark.__ab_hl');
if (first) first.scrollIntoView({block: 'center'});
return count;
"""

JS_PAGE_STATE = r"""
const out = [];
for (const e of document.querySelectorAll('a,button,input,select,textarea,[role=button],[role=link]')) {
  const r = e.getBoundingClientRect(); if (r.width < 2 || r.height < 2) continue;
  const cs = getComputedStyle(e); if (cs.visibility === 'hidden' || cs.display === 'none') continue;
  out.push([e.tagName.toLowerCase(), e.type || '', (e.innerText || e.textContent || e.value || '').trim().slice(0, 50),
            e.placeholder || '', e.getAttribute('aria-label') || '', e.name || '']);
  if (out.length >= 45) break;
}
return out;
"""

JS_LINKS = "return Array.from(document.querySelectorAll('a[href]')).map(a=>[(a.innerText || a.textContent || '').trim(), a.href]);"


class WebAgent:
    def __init__(self, log, ask_help, notes_file="agent_notes.json"):
        self.log = log
        self.ask_help = ask_help
        self.notes_file = notes_file
        self.driver = None
        self.attached = False
        self.lock = threading.RLock()

    def alive(self):
        if not self.driver:
            return False
        try:
            _ = self.driver.current_url
            return True
        except Exception:
            self.driver = None
            return False

    @staticmethod
    def _port_open(port):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                return True
        except OSError:
            return False

    def start(self):
        if not SELENIUM_OK:
            raise RuntimeError("Selenium is not installed. Run: pip install selenium")
        with self.lock:
            if self.alive():
                return
            if self._port_open(9222):
                try:
                    o = ChromeOptions()
                    o.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
                    self.driver, self.attached = webdriver.Chrome(options=o), True
                    self.log("Attached to your running Chrome (port 9222).")
                    return
                except Exception as e:
                    self.log(f"Could not attach to Chrome on 9222 ({e}); launching a new browser.")
            errs = []
            try:
                o = ChromeOptions()
                o.add_experimental_option("detach", True)
                o.add_argument("--start-maximized")
                self.driver, self.attached = webdriver.Chrome(options=o), False
                return
            except Exception as e:
                errs.append(f"Chrome: {str(e).splitlines()[0]}")
            try:
                self.driver, self.attached = webdriver.Firefox(options=FirefoxOptions()), False
                return
            except Exception as e:
                errs.append(f"Firefox: {str(e).splitlines()[0]}")
            raise RuntimeError("Could not start a browser. Install Chrome or Firefox.\n" + "\n".join(errs))

    def close(self):
        with self.lock:
            if not self.driver:
                return
            try:
                if self.attached:
                    self.driver.service.stop()
                else:
                    self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def _wait_loaded(self, timeout=15):
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete")
        except Exception:
            pass

    def _visible(self, selectors):
        for sel in selectors:
            try:
                for el in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if el.is_displayed() and el.is_enabled():
                        return el
            except Exception:
                continue
        return None

    @staticmethod
    def _xp_lower(expr):
        return f"translate({expr},'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')"

    def _click(self, el):
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.click()
        except Exception:
            self.driver.execute_script("arguments[0].click();", el)

    def open(self, url):
        self.start()
        with self.lock:
            target = normalize_url(url)
            self.driver.get(target)
            self._wait_loaded()
            return f"Opened {self.driver.current_url} (title: {self.driver.title!r})"

    def search(self, query):
        self.start()
        with self.lock:
            box = self._visible(SEARCH_BOX)
            if not box:
                trig = self._visible(SEARCH_TRIGGER)
                if trig:
                    self.log("No search box visible - clicking the search icon first.")
                    self._click(trig)
                    time.sleep(0.8)
                    box = self._visible(SEARCH_BOX)
            if not box:
                self.log("Could not find a search box - asking you for help.")
                ok = self.ask_help("I couldn't find the search box on this page.\n"
                                   "Please click into the site's search field, then press Done.")
                if not ok:
                    return "User cancelled; search not performed."
                box = self.driver.switch_to.active_element
            self._click(box)
            try:
                box.clear()
            except Exception:
                pass
            box.send_keys(query + Keys.ENTER)
            time.sleep(1.0)
            self._wait_loaded()
            return f"Searched for {query!r}. Now at {self.driver.current_url}"

    def click_text(self, text):
        self.start()
        with self.lock:
            low = text.lower().replace("'", "")
            xp = (f"//*[self::a or self::button or @role='button' or @role='link' or "
                  f"(self::input and (@type='submit' or @type='button'))]"
                  f"[contains({self._xp_lower('normalize-space(.)')},'{low}') or "
                  f"contains({self._xp_lower('@value')},'{low}') or "
                  f"contains({self._xp_lower('@aria-label')},'{low}') or "
                  f"contains({self._xp_lower('@title')},'{low}')]")
            for el in self.driver.find_elements(By.XPATH, xp):
                if el.is_displayed():
                    self._click(el)
                    time.sleep(0.8)
                    self._wait_loaded(8)
                    return f"Clicked element containing {text!r}. Now at {self.driver.current_url}"
            xp2 = f"//*[contains({self._xp_lower('normalize-space(text())')},'{low}')]"
            for el in self.driver.find_elements(By.XPATH, xp2):
                if el.is_displayed():
                    self._click(el)
                    time.sleep(0.8)
                    return f"Clicked text {text!r}."
            return f"No visible clickable element containing {text!r} was found."

    def type_into(self, field, text, submit=False):
        self.start()
        with self.lock:
            low = field.lower().replace("'", "")
            attrs = ["@placeholder", "@name", "@id", "@aria-label", "@autocomplete"]
            conds = " or ".join(f"contains({self._xp_lower(a)},'{low}')" for a in attrs)
            xp = (f"//input[not(@type='hidden') and not(@type='submit') and ({conds})] | "
                  f"//textarea[{conds}] | "
                  f"//label[contains({self._xp_lower('normalize-space(.)')},'{low}')]/following::input[1]")
            for el in self.driver.find_elements(By.XPATH, xp):
                if el.is_displayed() and el.is_enabled():
                    self._click(el)
                    try:
                        el.clear()
                    except Exception:
                        pass
                    el.send_keys(text + (Keys.ENTER if submit else ""))
                    return f"Typed into field matching {field!r}."
            return f"No visible input matching {field!r} was found."

    def press(self, key):
        self.start()
        k = getattr(Keys, key.upper(), None)
        if k is None:
            return f"Unknown key {key!r}."
        self.driver.switch_to.active_element.send_keys(k)
        return f"Pressed {key}."

    def highlight(self, phrase):
        self.start()
        with self.lock:
            n = self.driver.execute_script(JS_HIGHLIGHT, phrase)
            return int(n or 0)

    def page_state(self, text_chars=700):
        self.start()
        with self.lock:
            body = self.driver.find_element(By.TAG_NAME, "body").text
            els = self.driver.execute_script(JS_PAGE_STATE)
            lines = [f"URL: {self.driver.current_url}", f"Title: {self.driver.title}",
                     f"Visible text (start): {' '.join(body.split())[:text_chars]}",
                     "Interactive elements [tag,type,text,placeholder,aria-label,name]:"]
            lines += [f"  {json.dumps(e, ensure_ascii=False)}" for e in els]
            return "\n".join(lines)

    def _phones_on_page(self):
        body = self.driver.find_element(By.TAG_NAME, "body").text
        phones = extract_phones(body)
        for a in self.driver.find_elements(By.CSS_SELECTOR, "a[href^='tel:']"):
            num = unquote(a.get_attribute("href")[4:]).strip()
            if num and num not in phones:
                phones.append(num)
        for m in re.finditer(r'"telephone"\s*:\s*"([^"]+)"', self.driver.page_source):
            if m.group(1) not in phones:
                phones.append(m.group(1))
        return phones

    def find_phones(self, deep=True, max_extra_pages=3):
        self.start()
        with self.lock:
            home = self.driver.current_url
            results = {home: self._phones_on_page()}
            if deep:
                host = urlparse(home).netloc
                seen = {home}
                todo = []
                for text, href in self.driver.execute_script(JS_LINKS):
                    if (urlparse(href).netloc == host and href not in seen and
                            re.search(r"contact|about|support|reach|help|call|location", f"{text} {href}", re.I)):
                        seen.add(href)
                        todo.append(href)
                for href in todo[:max_extra_pages]:
                    try:
                        self.driver.get(href)
                        self._wait_loaded(10)
                        results[href] = self._phones_on_page()
                    except Exception as e:
                        self.log(f"Skipped {href}: {e}")
                self.driver.get(home)
                self._wait_loaded(10)
            results = {u: p for u, p in results.items() if p}
            for url, phones in results.items():
                self.save_note("phone_numbers", url, phones)
            if not results:
                return "No phone numbers found."
            return "Phone numbers found (saved to " + self.notes_file + "):\n" + \
                   "\n".join(f"  {u}: {', '.join(p)}" for u, p in results.items())

    def save_note(self, kind, url, data):
        try:
            try:
                with open(self.notes_file, "r", encoding="utf-8") as f:
                    notes = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                notes = []
            notes.append({"time": datetime.now().isoformat(timespec="seconds"),
                          "type": kind, "url": url, "data": data})
            with open(self.notes_file, "w", encoding="utf-8") as f:
                json.dump(notes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"Could not save note: {e}")