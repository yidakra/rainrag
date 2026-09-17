# Gateway origin

Browsers reach the Library at `https://rag.tvrain.io`, an external nginx TLS
gateway (Alexander Efimov, 172.16.52.75) that proxies to this box. The old name
`rag.tvrain.tv` returns 522: Cloudflare cannot reach the origin, which has no
public address and no tunnel.

`rag-gateway-origin.conf` holds both origin listeners. One origin carries every
path the browser asks for, because the player fetches media from the same host
as the page:

| Path | Upstream |
| --- | --- |
| `/` | Streamlit, 127.0.0.1:7861 |
| `/api/`, `/video/`, `/vtt/`, `/hls/` | API, 127.0.0.1:8001 |
| `/slack/` | Slack connector, 127.0.0.1:8002 |

## Port 8080, plain HTTP

Restricted to the gateway's source address. Being retired: it carries the login
form in cleartext across the office LAN, and a host on that segment can spoof a
source address.

## Port 8443, mutual TLS

The replacement. The gateway must present a certificate signed by the RainRAG
origin CA, so the hop is encrypted *and* the client is authenticated.

```bash
sudo bash deploy/nginx/origin-mtls-setup.sh init      # CA + origin cert, prints the CA to share
sudo bash deploy/nginx/origin-mtls-setup.sh sign gw.csr   # sign the gateway's CSR
```

**Never generate or transmit the gateway's private key.** The other side makes
a CSR, we return only the certificate.

What the gateway owner adds:

```nginx
proxy_pass https://172.16.52.220:8443;
proxy_ssl_certificate     /path/to/his.crt;
proxy_ssl_certificate_key /path/to/his.key;
proxy_ssl_trusted_certificate /path/to/rainrag-origin-ca.crt;
proxy_ssl_verify on;
proxy_ssl_name 172.16.52.220;
```

Verify before trusting it. A 200 with a valid certificate proves nothing on its
own; the point is that the other two cases fail:

```bash
curl --cert c.crt --key c.key --cacert ca.crt https://172.16.52.220:8443/_stcore/health   # ok
curl --cacert ca.crt https://172.16.52.220:8443/_stcore/health   # 400 No required SSL certificate
curl --cert other.crt --key other.key --cacert ca.crt https://172.16.52.220:8443/_stcore/health  # 400
```

Port 8080 is removed in its own change once the gateway is verified on 8443, so
the switch can be rolled back without the other side's help.
