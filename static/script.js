// static/script.js
async function formPost(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(data)
  });
  return res;
}

function setStatus(nextLoaded, speciesLoaded) {
  document.getElementById('mb-status').textContent = nextLoaded ? 'ready' : 'not loaded';
  document.getElementById('sp-status').textContent = speciesLoaded ? 'ready' : 'not loaded';
}

async function callRoute(button, route, payload, resultElemId) {
  button.disabled = true;
  try {
    const res = await formPost(route, payload);
    const data = await res.json();
    if (!res.ok) {
      document.getElementById(resultElemId).innerHTML = `<b>Error:</b> ${data.error || res.statusText}`;
      return;
    }
    if (data.next_base) {
      document.getElementById(resultElemId).innerHTML = `<b>Predicted Next Base:</b> <span class="big">${data.next_base}</span>`;
      setStatus(true, document.getElementById('sp-status').textContent === 'ready');
    } else if (data.species) {
      // show species and probabilities
      const species = data.species;
      let html = `<b>Species:</b> ${species}<div style="height:8px"></div>`;
      if (data.probabilities) {
        const items = Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]);
        for (const [lab, p] of items) {
          const pct = (p*100).toFixed(1);
          html += `<div class="prob-row"><div class="prob-label">${lab}</div>
                   <div class="prob-bar"><div class="prob-fill" style="width:${pct}%"></div></div>
                   <div class="prob-num">${pct}%</div></div>`;
        }
      }
      document.getElementById(resultElemId).innerHTML = html;
      setStatus(document.getElementById('mb-status').textContent === 'ready', true);
    } else {
      document.getElementById(resultElemId).innerHTML = `<b>Response:</b> ${JSON.stringify(data)}`;
    }
  } catch (err) {
    document.getElementById(resultElemId).innerHTML = `<b>Network/Error:</b> ${err.message}`;
  } finally {
    button.disabled = false;
  }
}

document.getElementById('predictBtn').addEventListener('click', async () => {
  const seq = document.getElementById('seqInput').value.trim();
  if (!seq) { document.getElementById('nextResult').innerHTML = '<b>Error:</b> Please paste a DNA sequence'; return; }
  await callRoute(document.getElementById('predictBtn'), '/predict', { sequence: seq }, 'nextResult');
});

document.getElementById('classifyBtn').addEventListener('click', async () => {
  const seq = document.getElementById('seqInput').value.trim();
  if (!seq) { document.getElementById('speciesResult').innerHTML = '<b>Error:</b> Please paste a DNA sequence'; return; }
  await callRoute(document.getElementById('classifyBtn'), '/classify', { sequence: seq }, 'speciesResult');
});

// On load, set initial status (will be updated once endpoints respond)
window.addEventListener('load', () => {
  // naive set: if element exists, set to ready if server-side models likely present
  setStatus(true, true);
});
