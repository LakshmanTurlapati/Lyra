#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank: technology topics (fresh for batch I)."""


def register(T):
    T["content delivery networks"] = {
        "_cat": "technology",
        "what": "geographically distributed networks of edge servers that cache and deliver web content (static assets, video, dynamic responses) close to end users, reducing latency and origin server load",
        "how": "DNS or anycast routes a user to a nearby edge server. The edge serves cached content directly; on cache miss it fetches from the origin (or a parent cache) and stores the result. Cache control headers, TTLs, and purging APIs govern freshness. Modern CDNs also run code at the edge for personalization and request routing",
        "why": "CDNs cut page load times, absorb traffic spikes, defend against DDoS attacks, reduce egress bandwidth costs, and improve availability. Most major sites rely on a CDN; performance and reliability of the open internet depend on the CDN layer",
        "vs": "a CDN differs from a single origin behind a load balancer by distributing copies geographically. It differs from a forward proxy (which serves clients out to the internet) and from a reverse proxy (which protects a single origin without distributed caching)",
        "ex": "Cloudflare's network terminates TLS and serves cached HTML and assets from over 300 cities, with origin pulls only on cache miss. A site can survive a viral traffic spike with origin load nearly flat because the CDN absorbs nearly all traffic",
        "mis": "people think a CDN automatically caches everything. By default it caches static assets, but dynamic content with cookies or auth headers is typically uncached unless the developer configures cache keys and TTLs explicitly",
    }

    T["zero-trust network architecture"] = {
        "_cat": "technology",
        "what": "a security model that treats every network request as untrusted regardless of origin, requiring per-request authentication, authorization, and continuous verification rather than relying on a perimeter firewall",
        "how": "every request authenticates with strong identity (often device certificate plus user MFA), is evaluated by a policy engine considering identity, device posture, and context, and is authorized to access only the specific resource. Microsegmentation prevents lateral movement; logging and analytics catch anomalies",
        "why": "zero trust addresses the failure of perimeter security in a world of cloud, remote work, and supply-chain attacks. It limits blast radius when credentials are stolen and is mandated by US federal cybersecurity directives. It is the dominant modern architecture pattern for enterprise security",
        "vs": "zero trust differs from castle-and-moat security, which trusts users inside the perimeter, and from VPN-based access, which extends a trusted network. It complements rather than replaces defense in depth (encryption, hardening, monitoring)",
        "ex": "Google's BeyondCorp eliminated VPNs by checking device identity and user context for every internal app request. An employee at a coffee shop has the same access experience as one in the office, with stronger granular controls",
        "mis": "people think zero trust means 'no trust anywhere'. It means trust is conditional and continuously verified, not absent. Another myth is that zero trust is a product; it is an architecture realized by combining identity, device, network, and application controls",
    }

    T["the OSI model"] = {
        "_cat": "technology",
        "what": "the seven-layer conceptual framework for networking developed by ISO that separates concerns from physical signaling at layer 1 to application protocols at layer 7, with each layer providing services to the one above and consuming services from the one below",
        "how": "layer 1 (physical) handles bits on wires; layer 2 (data link) frames and addresses on a local network; layer 3 (network) routes packets across networks via IP; layer 4 (transport) provides reliable streams via TCP or datagrams via UDP; layers 5-7 (session, presentation, application) handle higher-level protocol logic",
        "why": "the model gives a shared vocabulary for designing, debugging, and teaching networks. It clarifies which protocol or device handles which job (a switch operates at layer 2, a router at layer 3) and where to look when something breaks. It survives despite protocol stacks rarely matching it cleanly",
        "vs": "the OSI model differs from the TCP/IP model, which collapses to four or five layers and matches deployed protocols more directly. It differs from layered software architectures by being a network abstraction; the layering is purely conceptual at the wire level",
        "ex": "a packet from your browser to a server traverses HTTPS (layer 7) over TLS over TCP (layer 4) over IP (layer 3) framed by Ethernet (layer 2) on a copper or fiber medium (layer 1), with each device along the path operating at one or two layers",
        "mis": "people think OSI is implemented by all stacks. Real protocols (TCP/IP, HTTP, MPLS) cross or merge layers. The OSI model is a teaching scaffold, not a strict implementation specification",
    }

    T["service mesh architecture"] = {
        "_cat": "technology",
        "what": "an infrastructure layer that handles service-to-service communication in microservices architectures via sidecar proxies, providing traffic routing, observability, security, and reliability features without requiring application code changes",
        "how": "a sidecar proxy (Envoy, Linkerd) is deployed alongside each service instance. The proxy intercepts inbound and outbound traffic, applies policy from a control plane (Istio, Linkerd control), and emits metrics and traces. Mutual TLS, retries, circuit breakers, and traffic splitting happen in the proxy",
        "why": "service mesh centralizes cross-cutting concerns (security, observability, resilience) so application teams focus on business logic. It enables zero-trust networking, canary releases, and uniform telemetry across heterogeneous codebases, which became valuable as services proliferated to dozens or hundreds",
        "vs": "service mesh differs from API gateways, which sit at the edge handling external traffic. It differs from libraries embedded in each service (the older fat-client approach) by externalizing logic into proxies, enabling polyglot environments without per-language SDKs",
        "ex": "a Kubernetes cluster running Istio deploys an Envoy sidecar in every pod. The control plane configures mTLS between services, traffic shifts during deploys, and request tracing across the call graph, all without modifying application code",
        "mis": "people think service mesh is necessary for any microservices system. It adds significant operational complexity and resource overhead; small fleets often do better without one. The decision is about scale, security needs, and team capacity to operate it",
    }

    T["distributed consensus and Raft"] = {
        "_cat": "technology",
        "what": "the problem of getting multiple servers in a distributed system to agree on a sequence of values despite failures, and Raft, an algorithm designed to be more understandable than Paxos while providing the same guarantees",
        "how": "Raft elects a leader via randomized timeouts; the leader receives client commands, replicates them to followers, and commits an entry once a majority has acknowledged. If the leader fails, a new election begins. Log matching and election restrictions ensure safety: committed entries are never lost",
        "why": "consensus underpins distributed databases, configuration stores (etcd, Consul), and any system requiring linearizable state across nodes. Raft's clarity made it the basis for new systems and the dominant teaching algorithm, replacing Paxos in many production codebases",
        "vs": "Raft differs from Paxos by emphasizing understandability, with explicit leader election and log replication phases instead of Paxos's symmetric proposers. It differs from quorum-based eventually-consistent systems by providing strong consistency and linearizability",
        "ex": "etcd, used by Kubernetes for cluster state, runs Raft across three or five members. Cluster control plane operations (scheduling, configuration) survive any minority of node failures because the majority quorum continues to commit log entries",
        "mis": "people think Raft tolerates any failure. It tolerates up to floor((n-1)/2) failures with n members; lose more than that and progress halts. Another myth is that consensus avoids partitions; during partitions only the side with a majority makes progress",
    }

    T["GPUs and parallel computation"] = {
        "_cat": "technology",
        "what": "graphics processing units are massively parallel processors with thousands of simple cores optimized for throughput on data-parallel workloads, originally for graphics rendering and now central to scientific computing and machine learning",
        "how": "a GPU executes a single instruction across many threads in lockstep (SIMT). Threads are grouped into warps; warps share an instruction stream but operate on different data. Memory hierarchy (registers, shared memory, global memory) and high bandwidth feed the cores; programmers write kernels in CUDA, ROCm, or similar APIs",
        "why": "GPUs deliver one to two orders of magnitude more throughput than CPUs for matrix multiplications, convolutions, and other dense linear algebra at the heart of deep learning. They enabled the modern AI boom and dominate scientific simulations from molecular dynamics to fluid mechanics",
        "vs": "GPUs differ from CPUs by trading single-thread performance for parallel throughput, with shallower pipelines and simpler control. They differ from TPUs (tensor processing units), which specialize further in matrix and reduction primitives at the cost of flexibility",
        "ex": "training a large language model uses tens of thousands of GPUs (NVIDIA H100 or similar) connected by high-speed interconnects (NVLink, InfiniBand). Each GPU performs trillions of multiply-accumulate operations per second on weight tensors",
        "mis": "people think GPUs accelerate any program. They help only when the work is data-parallel; serial code or branch-heavy logic runs poorly. CPU-GPU data transfer is also a major bottleneck if not managed carefully",
    }

    T["public key infrastructure"] = {
        "_cat": "technology",
        "what": "the system of policies, hardware, software, and certificate authorities that issue, distribute, and validate digital certificates binding identities to public keys, enabling authentication and encryption at scale on the internet",
        "how": "a CA issues a certificate that binds an identity (domain, person) to a public key, signed by the CA's private key. Browsers and OSes ship a trust store of CA roots. When a client connects to a server, the server presents its certificate; the client verifies the chain back to a trusted root and uses the public key for the TLS handshake",
        "why": "PKI underlies HTTPS, code signing, email signing (S/MIME), VPN authentication, and document signing. Without it, web identity verification and large-scale encryption would not work. Certificate transparency logs and revocation systems support the trust model",
        "vs": "PKI differs from web-of-trust models like PGP, which rely on peer signatures rather than central CAs. It differs from symmetric key distribution, which requires shared secrets and does not scale to anonymous parties on the internet",
        "ex": "when you visit https://example.com, your browser receives a certificate signed by Let's Encrypt (a CA whose root is trusted by your browser), validates the chain, and uses the included public key to negotiate a TLS session that encrypts traffic",
        "mis": "people think a green padlock means the site is safe. It means the connection is encrypted to the named domain; phishing sites can also obtain valid certificates. Identity validation is only as strong as the CA's verification, which for domain-validated certs is automated",
    }

    T["WebRTC and real-time communication"] = {
        "_cat": "technology",
        "what": "an open standard and browser API enabling real-time audio, video, and data exchange between peers without plugins, used in video conferencing, telephony, gaming, and streaming",
        "how": "peers exchange signaling messages (out of band) to negotiate codecs and ICE candidates. ICE traverses NATs via STUN and TURN. DTLS-SRTP secures media; SCTP over DTLS carries data channels. Codecs (VP8, VP9, AV1, Opus) compress audio and video; jitter buffers and FEC handle network variation",
        "why": "WebRTC enables real-time communication directly in browsers and native apps with predictable APIs. Zoom, Google Meet, Discord, and many telehealth platforms use WebRTC for parts of their pipelines. It commoditized peer-to-peer media transport that previously required custom stacks",
        "vs": "WebRTC differs from RTMP/HLS streaming, which is one-to-many and tolerates latency of seconds. It differs from SIP-based VoIP by being browser-native and built around modern security defaults. It differs from raw UDP applications by including NAT traversal and congestion control",
        "ex": "Google Meet uses WebRTC end-to-end with a media server (SFU) that selectively forwards streams. The browser handles encoding, decryption, and rendering; the SFU routes media without re-encoding, enabling group calls with hundreds of participants on commodity hardware",
        "mis": "people think WebRTC is fully peer-to-peer. For two users it can be, but multi-party calls almost always use a media server (SFU or MCU) for fan-out and recording. Pure mesh topology breaks down beyond a handful of participants",
    }

    T["the Linux process model"] = {
        "_cat": "technology",
        "what": "the abstraction by which Linux gives each running program its own address space, file descriptor table, and execution context, with processes spawned via fork and replaced via exec, and isolated by user, namespace, and cgroup",
        "how": "fork() duplicates the calling process; the child can then exec() a new program. Each process has a PID, parent PID, and unique virtual address space. The kernel scheduler picks ready processes by policy (CFS for normal tasks, real-time classes for time-critical work). signals and pipes allow inter-process communication",
        "why": "the process model isolates failures and provides resource accounting. It is the basis for shells, services, container runtimes, and security models. Containers extend the model with namespaces (PID, network, mount) and cgroups for resource limits, all building on the basic process abstraction",
        "vs": "Linux processes differ from threads, which share an address space and file descriptors within a process. They differ from Windows process creation (CreateProcess), which combines fork-and-exec into a single call and never duplicates address spaces",
        "ex": "running a program at the shell calls fork() to create a child, then exec() to load the program image. The original shell process resumes after the child exits. ps and /proc expose process state for inspection",
        "mis": "people think fork() is expensive because it copies memory. Modern Linux uses copy-on-write, so the duplication is virtual; pages are only physically copied when one side writes. fork+exec without much in-between is cheap",
    }

    T["Bloom filters"] = {
        "_cat": "technology",
        "what": "a space-efficient probabilistic data structure that tests whether an element is in a set, with no false negatives but a tunable false-positive rate, achieving compact representation at the cost of certainty",
        "how": "k hash functions map an element to k positions in a bit array of size m; insertion sets those bits, lookup checks them. If any is zero the element is definitely absent; if all are one it is probably present. Optimal k depends on m and expected element count n, balancing collision probability",
        "why": "Bloom filters reduce expensive lookups in large datasets: a database checks the filter before disk I/O for a possibly-absent key. They are used in caches, content-based routing, distributed systems, and bioinformatics for k-mer counting in massive sequence data",
        "vs": "Bloom filters differ from hash sets by allowing false positives in exchange for far less memory. They differ from cuckoo filters, which support deletion and slightly better space efficiency, and from count-min sketches, which estimate frequencies rather than membership",
        "ex": "Google Chrome historically used a Bloom filter to check URLs against a malicious-site list locally; only on a positive hit did it consult the network. The filter avoided sending every navigation to a remote service while preserving safety against false negatives",
        "mis": "people think Bloom filters give exact membership. The 'present' answer is probabilistic; downstream code must handle false positives, often by following up with an authoritative source. The 'absent' answer is exact",
    }

    T["just-in-time compilation in V8"] = {
        "_cat": "technology",
        "what": "the layered compilation pipeline used by Google's V8 JavaScript engine that interprets code initially, then compiles hot functions to optimized machine code with speculative type assumptions, and deoptimizes when assumptions fail",
        "how": "V8 parses JavaScript to bytecode for the Ignition interpreter. Hot functions are tiered up to TurboFan or Maglev, which use type feedback to generate specialized machine code. If a runtime check disproves an assumption (e.g., expecting an integer but seeing a string), V8 deoptimizes back to bytecode and may re-optimize later",
        "why": "JIT compilation lets dynamic languages approach the speed of statically compiled ones. V8's design powers Chrome and Node.js, making JavaScript viable as a server runtime. The engineering pattern (interpret, profile, specialize, deoptimize) shapes modern dynamic-language performance",
        "vs": "JIT differs from ahead-of-time compilation, which produces machine code before execution and cannot specialize on runtime values. It differs from pure interpretation by paying compilation cost to recover at runtime, with profile-driven decisions about what to compile",
        "ex": "a JavaScript function that always receives integers will be specialized by TurboFan to use integer arithmetic. If the same function is later called with a string, V8 deoptimizes the integer version and recompiles a more general one, with measurable but small latency",
        "mis": "people think 'JavaScript is slow because it's interpreted'. Modern engines are JIT compilers; well-typed JavaScript can run within 2-3x of optimized C for numeric code. Performance pitfalls usually come from hidden type changes, megamorphic property accesses, or suboptimal data layouts",
    }

    T["columnar database storage"] = {
        "_cat": "technology",
        "what": "a database storage layout that stores values of each column contiguously rather than each row contiguously, dramatically improving compression and analytical query performance at the cost of more expensive single-row writes",
        "how": "the storage engine writes column files (or column chunks within blocks). Queries that touch only a few columns read just those, skipping irrelevant data. Run-length, dictionary, delta, and bit-packing compression apply well to similar values within a column. Vectorized execution batches values for SIMD-friendly processing",
        "why": "columnar layout is the foundation of modern analytics warehouses (BigQuery, Snowflake, Redshift, ClickHouse, DuckDB) and on-disk formats (Parquet, ORC). It speeds aggregations and scans on wide tables by orders of magnitude compared to row stores, and compresses data 5-10x",
        "vs": "columnar storage differs from row stores (PostgreSQL, MySQL) which favor OLTP workloads with point lookups and small updates. Hybrid approaches (HTAP systems, hybrid layouts) try to bridge both. PAX layouts blend row and column ideas within a page",
        "ex": "a query computing daily revenue from a 200-column orders table reads only the date and amount columns, perhaps 1 percent of the table's data. The same query on a row store would touch every byte of every row scanned",
        "mis": "people think columnar is universally better. For high-throughput single-row inserts, updates, and lookups, row stores still win. The choice depends on workload: columnar shines on append-heavy analytics, row stores on transactional applications",
    }

    T["the actor model in Erlang"] = {
        "_cat": "technology",
        "what": "a concurrency model in which the basic unit is the actor, an isolated entity with private state that communicates only by passing immutable messages, used by Erlang and the BEAM virtual machine to build highly available distributed systems",
        "how": "the BEAM creates lightweight processes (actors), each with its own heap and message queue. Processes send messages with !, receive selectively, and supervise children. If a child crashes, supervisors restart it; failure is normal and recovery is automatic. Hot code reload swaps modules without stopping the system",
        "why": "Erlang's actor model produced systems with famous availability records, including Ericsson's AXD301 telephone switch (nine 9s of uptime). It influenced WhatsApp, Discord, and Elixir, and the let-it-crash philosophy reshaped how engineers think about fault tolerance",
        "vs": "the actor model differs from shared-memory threading, which requires locks and risks data races. It differs from CSP-style channels (Go), which are between actors but more synchronous. It differs from microservices by being in-process and message-passing through a runtime",
        "ex": "WhatsApp scaled to a billion users with a small engineering team using Erlang. Each connection was a process; supervisors restarted failures; and the system handled massive concurrency with minimal locking and predictable degradation under load",
        "mis": "people think actors are slow because they pass messages. BEAM message passing within a node is on the order of microseconds; the model trades a bit of latency for huge isolation and reliability gains. Another myth is that the actor model only works in Erlang; Akka, Pony, and others adapted it",
    }

    T["progressive web apps"] = {
        "_cat": "technology",
        "what": "web applications that use modern browser APIs (service workers, manifests, push notifications) to deliver app-like experiences such as offline use, installable home screen icons, and background sync without an app store",
        "how": "a service worker, a JavaScript thread separate from pages, intercepts network requests and serves cached responses for offline use. A web app manifest declares icons, theme colors, and start URL. Push API and Web Notification API support engagement; IndexedDB stores structured data offline",
        "why": "PWAs reduce engineering and distribution costs by reusing web codebases while reaching capabilities once exclusive to native apps. They sidestep app stores' gatekeeping and 30 percent fees, work on any modern browser, and update instantly without user intervention",
        "vs": "PWAs differ from native apps, which are platform-specific binaries with deeper OS integration. They differ from hybrid apps (Cordova, Capacitor) that wrap web in a native shell. They differ from regular web pages by being installable and offline-capable",
        "ex": "Twitter Lite, a PWA, reduced page load times and data usage on slow networks while supporting installation. Starbucks rebuilt their ordering experience as a PWA, gaining offline menu browsing and faster startup compared to the previous web app",
        "mis": "people think PWAs are second-class citizens that cannot do what native does. On Android, the gap is small; on iOS, Apple historically limited some APIs but support has improved. PWAs may suit many use cases that previously required native, particularly content-heavy apps",
    }
