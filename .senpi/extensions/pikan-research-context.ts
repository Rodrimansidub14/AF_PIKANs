import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { promises as fs } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { spawn } from "node:child_process";

import {
	DEFAULT_ROLE,
	filesystemDecision,
	renderSystemPrompt,
	selectWorkflow,
} from "./pikan-research-context-core.mjs";

const WATCH_PATHS = [
	"context",
	"workflow",
	"PIKAN-Vault/PIKAN-Vault",
	"notebooks",
	"src/harness_loop/experiment_manifest.yaml",
];
const MAX_PROCESS_MS = 60_000;

function findProjectRoot(startPath) {
	let current = resolve(startPath);
	while (true) {
		if (
			existsSync(join(current, "context", "agent-policy.toml")) &&
			existsSync(join(current, "src", "graph_engineering"))
		) {
			return current;
		}
		const parent = dirname(current);
		if (parent === current) return undefined;
		current = parent;
	}
}

function errorMessage(error) {
	return error instanceof Error ? error.message : String(error);
}

function runProcess(command, args, cwd) {
	return new Promise((resolveProcess, rejectProcess) => {
		const child = spawn(command, args, {
			cwd,
			windowsHide: true,
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout = "";
		let stderr = "";
		const timer = setTimeout(() => {
			child.kill();
			rejectProcess(new Error(`${command} timed out after ${MAX_PROCESS_MS}ms`));
		}, MAX_PROCESS_MS);

		child.stdout?.on("data", (chunk) => { stdout += chunk.toString(); });
		child.stderr?.on("data", (chunk) => { stderr += chunk.toString(); });
		child.on("error", (error) => {
			clearTimeout(timer);
			if (error?.code === "ENOENT") {
				rejectProcess(Object.assign(new Error(`${command} was not found`), { code: "ENOENT" }));
				return;
			}
			rejectProcess(error);
		});
		child.on("close", (code) => {
			clearTimeout(timer);
			if (code === 0) {
				resolveProcess({ stdout, stderr });
				return;
			}
			rejectProcess(new Error(`${command} exited with ${code}: ${stderr || stdout}`.trim()));
		});
	});
}

async function runPython(args, cwd) {
	const candidates = process.platform === "win32"
		? [
			{ command: "py", prefix: ["-3.12"] },
			{ command: "python", prefix: [] },
			{ command: "python3", prefix: [] },
		]
		: [
			{ command: "python3", prefix: [] },
			{ command: "python", prefix: [] },
		];
	let lastError;
	for (const candidate of candidates) {
		try {
			return await runProcess(candidate.command, [...candidate.prefix, ...args], cwd);
		} catch (error) {
			lastError = error;
			if (error?.code !== "ENOENT") throw error;
		}
	}
	throw lastError ?? new Error("No Python interpreter was found.");
}

async function listFiles(root) {
	if (!existsSync(root)) return [];
	const entries = await fs.readdir(root, { withFileTypes: true });
	const files = [];
	for (const entry of entries) {
		const path = join(root, entry.name);
		if (entry.isDirectory()) files.push(...await listFiles(path));
		else if (entry.isFile()) files.push(path);
	}
	return files;
}

async function sourceFingerprint(projectRoot) {
	const hash = createHash("sha256");
	for (const watchedPath of WATCH_PATHS) {
		const absolute = join(projectRoot, watchedPath);
		const paths = existsSync(absolute) && (await fs.stat(absolute)).isDirectory()
			? await listFiles(absolute)
			: existsSync(absolute) ? [absolute] : [];
		for (const filePath of paths.sort()) {
			const stat = await fs.stat(filePath);
			hash.update(relative(projectRoot, filePath).split(sep).join("/"));
			hash.update(`:${stat.size}:${stat.mtimeMs};`);
		}
	}
	return hash.digest("hex");
}

async function readJson(path) {
	return JSON.parse(await fs.readFile(path, "utf8"));
}

function emptyPack() {
	return {
		ontology: "pikan-research",
		roles: [{ id: DEFAULT_ROLE, purpose: "Conservative read-only research role.", read_paths: ["context"], write_paths: [] }],
		gates: [],
		graph: { nodes: 0, edges: 0 },
		notebooks: { count: 0 },
		workflow: {
			default: "literature",
			ambiguity_policy: "conservative",
			items: [{
				id: "literature",
				role: DEFAULT_ROLE,
				priority: 0,
				triggers: [],
				excludes: [],
				stages: ["orient"],
				gate_ids: [],
			}],
		},
	};
}

export default function pikanResearchContext(pi) {
	const projectRoot = findProjectRoot(process.cwd());
	const state = {
		root: projectRoot,
		pack: undefined,
		graph: undefined,
		fingerprint: undefined,
		ready: false,
		error: undefined,
		roleId: DEFAULT_ROLE,
		directiveError: undefined,
		preparing: undefined,
	};

	const protectedRoots = projectRoot
		? [
			join(projectRoot, "context"),
			join(projectRoot, "PIKAN-Vault", "PIKAN-Vault"),
			join(projectRoot, ".senpi"),
			join(projectRoot, "var", "context"),
		]
		: [];

	pi.registerFilesystemPolicy({
		deniedRoots: protectedRoots,
		check({ operation, canonicalPath }) {
			if (!state.root || !state.pack) {
				return { allow: false, reason: "Research context has not been validated for this project." };
			}
			if (!state.ready && operation === "write") {
				return { allow: false, reason: "Research context validation failed; writes are disabled." };
			}
			if (state.directiveError && operation === "write") {
				return { allow: false, reason: state.directiveError };
			}
			const role = state.pack.roles.find((item) => item.id === state.roleId)
				?? state.pack.roles.find((item) => item.id === DEFAULT_ROLE);
			return filesystemDecision({
				root: state.root,
				role,
				operation,
				canonicalPath,
			});
		},
	});

	async function ensureContext() {
		if (!state.root) throw new Error("ProyectoGrad root was not found from the current working directory.");
		if (state.preparing) return state.preparing;
		state.preparing = (async () => {
			const fingerprint = await sourceFingerprint(state.root);
			const graphPath = join(state.root, "var", "context", "research-graph.json");
			const packPath = join(state.root, "var", "context", "senpi-context.json");
			if (state.fingerprint === fingerprint && state.graph && state.pack && existsSync(graphPath) && existsSync(packPath)) {
				return { pack: state.pack, graph: state.graph };
			}

			await runPython(["-m", "src.graph_engineering", "validate", "--root", "."], state.root);
			await runPython(["src/research_pipeline/pipeline.py", "validate", "--manifest", "src/harness_loop/experiment_manifest.yaml"], state.root);
			await runPython(["-m", "src.graph_engineering", "build", "--root", ".", "--out", "var/context"], state.root);
			const [pack, graph] = await Promise.all([readJson(packPath), readJson(graphPath)]);
			state.fingerprint = fingerprint;
			state.pack = pack;
			state.graph = graph;
			state.ready = true;
			state.error = undefined;
			return { pack, graph };
		})().catch((error) => {
			state.ready = false;
			state.error = errorMessage(error);
			throw error;
		}).finally(() => {
			state.preparing = undefined;
		});
		return state.preparing;
	}

	function notify(ctx, message, type = "info") {
		ctx.ui?.notify?.(message, type);
	}

	pi.registerCommand("research-context", {
		description: "Validate and show the PIKAN research context",
		handler: async (_args, ctx) => {
			try {
				const result = await ensureContext();
				ctx.ui?.setStatus?.("research-context", `ready: ${result.pack.graph.nodes} nodes`);
				notify(ctx, `Research context ready: ${result.pack.graph.nodes} nodes, ${result.pack.graph.edges} edges.`, "info");
			} catch (error) {
				notify(ctx, `Research context unavailable: ${errorMessage(error)}`, "error");
			}
		},
	});

	pi.on("session_start", async (_event, ctx) => {
		try {
			const result = await ensureContext();
			ctx.ui?.setStatus?.("research-context", `ready: ${result.pack.graph.nodes} nodes`);
		} catch (error) {
			ctx.ui?.setStatus?.("research-context", "blocked: validation failed");
			notify(ctx, `Research context blocked: ${errorMessage(error)}`, "error");
		}
	});

	pi.on("before_agent_start", async (event, ctx) => {
		let pack = state.pack ?? emptyPack();
		let graph = state.graph ?? { nodes: [], edges: [], diagnostics: {}, notebook_audit: {} };
		try {
			const result = await ensureContext();
			pack = result.pack;
			graph = result.graph;
		} catch (error) {
			state.error = errorMessage(error);
		}

		const routing = selectWorkflow(event.prompt, pack.workflow, pack.roles ?? []);
		state.roleId = routing.role?.id ?? DEFAULT_ROLE;
		state.directiveError = routing.error;
		const role = routing.role ?? pack.roles?.find((item) => item.id === DEFAULT_ROLE) ?? emptyPack().roles[0];
		const context = renderSystemPrompt({
			pack,
			graph,
			role,
			workflow: routing.workflow,
			prompt: event.prompt,
			ready: state.ready && !routing.error,
			error: routing.error ?? state.error,
		});
		return { systemPrompt: `${event.systemPrompt}\n\n${context}` };
	});

	pi.on("tool_call", async (event) => {
		if (event.toolName !== "bash") return undefined;
		const command = String(event.input?.command ?? "");
		const normalized = command.toLowerCase();
		if (normalized.includes("obs-data")) {
			return { block: true, reason: "Bash access to obs-data is forbidden for this project." };
		}
		const protectedTarget = ["pikan-vault", "context/", "context\\", "workflow/", "workflow\\", ".senpi", ".git"].some((token) => normalized.includes(token));
		const mutation = /(^|\s)(rm|rm -rf|del|erase|mv|move|cp|copy|remove-item|move-item|copy-item|set-content|out-file|git\s+(reset|clean|checkout|apply|commit))\b|[>]{1,2}/i.test(command);
		if (protectedTarget && mutation) {
			return { block: true, reason: "Bash mutation of protected research-control paths is forbidden." };
		}
		return undefined;
	});
}
