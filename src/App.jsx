import { useState, useEffect, useRef } from "react";

// IndexedDB helpers
const DB_NAME = "LLMDB";
const STORE_NAME = "kvCache";

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE_NAME);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function putKV(layer, kv) {
  return openDB().then(db => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(kv, `layer_${layer}`);
    return tx.complete;
  });
}

function getKV(layer) {
  return openDB().then(db => new Promise(resolve => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(`layer_${layer}`);
    req.onsuccess = () => resolve(req.result || { keys: [], values: [] });
    req.onerror = () => resolve({ keys: [], values: [] });
  }));
}

// --- Tiny transformer parameters ---
const d_model = 32;
const n_layers = 2;
const vocab_size = 100; // small toy vocab
const EOS = 99;        // last token is EOS
const maxTokens = 100;  // maximum tokens per response

function randomMatrix(rows, cols) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => Math.random() * 0.2 - 0.1)
  );
}

const Wq = Array.from({ length: n_layers }, () => randomMatrix(d_model, d_model));
const Wk = Array.from({ length: n_layers }, () => randomMatrix(d_model, d_model));
const Wv = Array.from({ length: n_layers }, () => randomMatrix(d_model, d_model));
const Wo = Array.from({ length: n_layers }, () => randomMatrix(d_model, d_model));
const W_embed = randomMatrix(vocab_size, d_model);
const W_out = randomMatrix(d_model, vocab_size);

// --- Math helpers ---
function matMul(A, B) {
  const rows = A.length, cols = B[0].length, K = B.length;
  const out = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      for (let k = 0; k < K; k++)
        out[i][j] += A[i][k] * B[k][j];
  return out;
}

function softmax(vec) {
  const max = Math.max(...vec);
  const exps = vec.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// --- Tiny BPE-like tokenizer ---
function encode(str) {
  return Array.from(str).map(c => c.charCodeAt(0) % (vocab_size - 1)); // map to 0..vocab-2
}
function decode(arr) {
  return arr.map(x => (x === EOS ? "" : String.fromCharCode(x + 65))).join(""); // simple demo
}

// --- React App ---
export default function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const kvCacheRef = useRef({});
  const messagesEndRef = useRef(null);

  useEffect(() => {
    (async () => {
      for (let l = 0; l < n_layers; l++) {
        kvCacheRef.current[l] = await getKV(l);
      }
    })();
  }, []);

  const transformerStep = async (inputToken) => {
    let x = W_embed[inputToken];
    for (let l = 0; l < n_layers; l++) {
      const kv = kvCacheRef.current[l] || { keys: [], values: [] };
      const q = matMul([x], Wq[l])[0];
      const k = matMul([x], Wk[l])[0];
      const v = matMul([x], Wv[l])[0];
      kv.keys.push(k);
      kv.values.push(v);

      const scores = kv.keys.map(kvk => kvk.reduce((a, b, i) => a + kvk[i] * q[i], 0));
      const probs = softmax(scores);
      let attn = Array(d_model).fill(0);
      for (let i = 0; i < kv.values.length; i++)
        for (let j = 0; j < d_model; j++)
          attn[j] += kv.values[i][j] * probs[i];

      x = matMul([attn], Wo[l])[0];
      kvCacheRef.current[l] = kv;
      putKV(l, kv);
    }

    const logits = matMul([x], W_out)[0];
    const maxIdx = logits.indexOf(Math.max(...logits));
    return maxIdx;
  };

  const generateBotResponse = async (userTokens) => {
    let outputTokens = [];
    let inputToken = userTokens[0]; // first token
    for (let i = 0; i < maxTokens; i++) {
      const tok = await transformerStep(inputToken);
      if (tok === EOS) break;
      outputTokens.push(tok);
      inputToken = tok;
    }
    return outputTokens;
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = input;
    setMessages(prev => [...prev, { role: "user", text: userMessage }]);
    setInput("");

    const userTokens = encode(userMessage);
    const botTokens = await generateBotResponse(userTokens);

    // Typing effect
    setMessages(prev => [...prev, { role: "bot", text: "" }]);
    for (let i = 0; i < botTokens.length; i++) {
      await new Promise(r => setTimeout(r, 50));
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1].text += decode([botTokens[i]]);
        return newMessages;
      });
      scrollToBottom();
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      height: "100vh", maxWidth: "600px", margin: "0 auto",
      border: "1px solid #ccc", borderRadius: "10px", overflow: "hidden"
    }}>
      <h1>Browser GPT</h1>
      <p>Untrained model, do not expect coherent output.</p>
      <div style={{
        flex: 1, padding: "1rem", overflowY: "auto", background: "#f5f5f5"
      }}>
        {messages.map((m, i) => (
          <div key={i} style={{
            display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start",
            marginBottom: "0.5rem"
          }}>
            <div style={{
              maxWidth: "70%", padding: "0.5rem 1rem",
              background: m.role === "user" ? "#007bff" : "#e0e0e0",
              color: m.role === "user" ? "#fff" : "#000",
              borderRadius: "15px", wordBreak: "break-word"
            }}>
              {m.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div style={{ display: "flex", borderTop: "1px solid #ccc" }}>
        <textarea
          rows={2}
          style={{ flex: 1, border: "none", padding: "0.5rem" }}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder="Type a message..."
        />
        <button onClick={handleSend} style={{ width: "80px", border: "none", background: "#007bff", color: "#fff" }}>
          Send
        </button>
      </div>
    </div>
  );
}
