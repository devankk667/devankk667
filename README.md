<div align="center">

# `> hello, world_`

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=22&duration=3000&pause=1000&color=00E5A0&center=true&vCenter=true&width=700&lines=I'm+Devankk667;Student+%2B+Researcher;Spatial+GroupKFold+%3E+random+split;Tool-calling+LLMs+%2B+RAG+%2B+geospatial+ML;Building+ML+systems+that+ship+%F0%9F%9A%80)](https://github.com/devankk667)

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/-React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

---

## 🖥️ `whoami`

```text
devankk667@github
-----------------
role      : Student & Researcher
focus     : Applied ML · Geospatial ML · LLM-powered apps · Full-stack
languages : Python, JavaScript
serving   : FastAPI · Express 5
learning  : tool-calling agents, RAG, leakage-safe model evaluation
status    : open to internships → ML Engineer | Data Scientist | Data Analyst
```

```python
class Devankk667:
    mantra = "Transform data into actionable insights."

    def build(self, problem):
        data     = self.clean(problem)                  # garbage in, garbage out
        features = self.engineer(data)                  # domain knowledge > bigger model
        model    = self.validate_without_leakage(features)  # the part people skip
        return self.ship(model)                         # API + UI + Docker, not just a notebook
```

---

## 🧰 Tech Stack (by layer)

| Layer | Tools |
|---|---|
| **Languages** | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) ![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black) |
| **Data & Classical ML** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/-XGBoost-189AB4?style=flat-square) |
| **Deep Learning & NLP** | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![spaCy](https://img.shields.io/badge/-spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white) `sentence-transformers` `Whisper` |
| **Backend** | ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Flask](https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white) ![Django](https://img.shields.io/badge/-Django-092E20?style=flat-square&logo=django&logoColor=white) ![Node.js](https://img.shields.io/badge/-Node.js-339933?style=flat-square&logo=nodedotjs&logoColor=white) ![Express](https://img.shields.io/badge/-Express-000000?style=flat-square&logo=express&logoColor=white) |
| **Frontend** | ![React](https://img.shields.io/badge/-React-20232A?style=flat-square&logo=react&logoColor=61DAFB) ![Vite](https://img.shields.io/badge/-Vite-646CFF?style=flat-square&logo=vite&logoColor=white) ![Expo](https://img.shields.io/badge/-Expo-000020?style=flat-square&logo=expo&logoColor=white) `Leaflet` `Recharts` |
| **Storage** | ![MongoDB](https://img.shields.io/badge/-MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white) ![SQLite](https://img.shields.io/badge/-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white) |
| **DevOps** | ![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white) `docker-compose` `pytest` |

---

## 🌟 Featured Projects

### 🔥 [namyadam](https://github.com/devankk667/namyadam): AI Industrial Thermal Anomaly & Fire Monitoring

> **Problem:** satellites see heat, not intent. A refinery flare stack, a wildfire, and a crop burn all look like "a hot pixel". This system uses thermal physics plus spatial context to tell them apart.

**How it works**

```mermaid
flowchart LR
    A["NASA FIRMS thermal detections"] --> C["Feature engineering"]
    B["OpenStreetMap infrastructure proximity"] --> C
    C --> D["ML engine: LogReg / RF / XGBoost / PyTorch NN"]
    D --> E["FastAPI service"]
    E --> F["Alert rule engine"]
    E --> G["React + Leaflet dashboard"]
```

**Engineering highlights**

- **Physics-informed features:** uses brightness-temperature difference `T_diff = T4 − T31` and fire radiative power (FRP) as signals, rather than only raw pixels.
- **Spatial context:** OSM proximity features separate "persistent industrial source" from "transient wildfire".
- **Leakage-safe validation:** a random split lets nearby hotspots from the *same* industrial cluster land in both train and test, which inflates scores. Spatial `GroupKFold` keeps whole clusters together so the metrics are honest.
- **Model bake-off:** Logistic Regression → Random Forest → XGBoost → PyTorch NN, compared on the same grouped splits.
- **Early-warning engine:** rule-based alerts for high-FRP outbursts, industrial fires, and persistent flare anomalies.
- **Interactive prediction studio:** feed custom feature vectors to trained checkpoints and see feature importances.
- **Offline demo mode:** `DEMO_MODE=true` runs on a bundled dataset centred on real industrial hubs (Permian Basin, Ruhr Valley, Houston Ship Channel, Jurong Island), so it works with no API keys.

<details>
<summary><b>📡 API surface (click to expand)</b></summary>

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/detections` | GET | Paginated detections, filter by confidence / FRP / region |
| `/api/detections/{id}` | GET | Event detail with spatial context and live inference |
| `/api/predictions` | POST | Classify a custom input vector |
| `/api/analytics/{summary,temporal,classification,regions,persistence}` | GET | Metrics, trends, regional density, recurring flare stacks |
| `/api/alerts` | GET | Alerts from the risk engine |
| `/api/models` | GET | Benchmarks, feature importances, version metadata |

</details>

```bash
docker-compose up --build        # API :8000 (Swagger at /docs) · dashboard :5173
```

`Python 3.12` `FastAPI` `PyTorch` `XGBoost` `scikit-learn` `React 18` `Vite` `Leaflet` `Recharts` `Docker` `pytest`

---

### 🎓 [DevOops](https://github.com/devankk667/DevOops): AetherOS, an AI-powered campus platform

*Contributor* · forked from [divykasodariya/DevOops](https://github.com/divykasodariya/DevOops)

> **Problem:** campus workflows (attendance, leave approvals, bookings) are scattered across forms and WhatsApp groups. AetherOS puts them in one role-aware mobile app with an AI copilot on top.

**Three-tier architecture**

```mermaid
flowchart TB
    M["Mobile client: React Native + Expo SDK 54"] -->|"REST + JWT"| N["Core API: Node.js ESM + Express 5 + MongoDB"]
    M -->|"text / voice"| A["AI microservice: FastAPI + LLM via Groq"]
    A -->|"tool calls, proxied with the user's JWT"| N
```

**Engineering highlights**

- **Permission-safe AI agent:** the copilot never gets a privileged key. The AI service forwards the *user's own JWT* back to the Node API when executing tool calls, so the LLM can only do what that user could already do.
- **Role-based access control:** separate permissions for Students, Faculty, HODs, Principals, and Admins.
- **Idempotent attendance:** course-linked, so duplicate submissions can't double-count.
- **Multi-tier approval routing:** leave requests, recommendation letters, and facility bookings flow through configurable approval chains.
- **Voice interface:** on-device recording with Whisper transcription feeds the same assistant pipeline.

`React 19` `React Native` `Expo Router` `Node.js` `Express 5` `MongoDB` `Mongoose 9` `JWT` `FastAPI` `Groq` `Whisper`

---

### 📬 [email_curator](https://github.com/devankk667/email_curator): Mail Curator.ai

> **Problem:** inboxes are unstructured text with hidden deadlines. This app classifies, ranks, searches, and summarizes mail, mostly with local models.

**Processing pipeline**

```mermaid
flowchart LR
    A[".eml / Gmail OAuth"] --> B["Loader + cleaner"]
    B --> C["Features: TF-IDF or embeddings + sender-domain signals"]
    C --> D["Logistic Regression: 8 categories"]
    C --> E["Semantic priority scoring"]
    B --> F["Local NLP: spaCy + regex"]
    D --> G[("SQLite cache")]
    E --> G
    F --> G
    G --> H["FastAPI + dashboard"]
```

**Engineering highlights**

- **Hybrid training data:** synthetic emails + SpamAssassin corpus + real Gmail messages, retrainable from the UI via a background job.
- **Richer than bag-of-words:** `sentence-transformers` embeddings combined with engineered features and sender-domain signals.
- **Local intelligence engine:** extractive summaries, action items, deadlines, and entities (orgs, locations, money, reference numbers) via spaCy plus context rules.
- **RAG-style chat:** retrieves relevant emails by semantic similarity, then answers with an optional Groq LLM or a local fallback.
- **SQLite metadata cache:** avoids recomputing expensive NLP on every request.
- **Deployment-ready:** Docker, docker-compose, and Vercel/Render config.

<details>
<summary><b>🧪 Engineering notes: what I'd improve next (click to expand)</b></summary>

- The training set is mostly synthetic, so I'd build a small manually labelled real-inbox validation set for a trustworthy accuracy number.
- Search and chat recompute embeddings per request, so I'd precompute and persist them.
- I'd add API and retraining tests beyond the current cleaner tests.

</details>

`Python` `FastAPI` `scikit-learn` `sentence-transformers` `spaCy` `SQLite` `Gmail API` `Docker`

---

## 📊 GitHub Stats

<div align="center">

![Devankk667's GitHub stats](https://github-readme-stats.vercel.app/api?username=devankk667&show_icons=true&theme=tokyonight&hide_border=true)
![Top languages](https://github-readme-stats.vercel.app/api/top-langs/?username=devankk667&layout=compact&theme=tokyonight&hide_border=true)

</div>

---

## 🎯 Career Goals

```bash
$ cat goals.txt
> Internship as an ML Engineer, Data Scientist, or Data Analyst
> Ship ML systems that solve real problems, not just notebooks
> Join a team doing impactful work

$ git log --oneline --author=devankk667   # still committing
```

---

## 📡 Connect

```bash
$ sudo hire devankk667
[sudo] password for recruiter: ********
Access granted. 🎉
```

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devank-kolpe/)
