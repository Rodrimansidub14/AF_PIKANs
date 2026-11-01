import { isAbsolute, relative, resolve, sep } from "node:path";

export const DEFAULT_ROLE = "literature_scout";

function normalizeText(value) {
	return String(value ?? "")
		.normalize("NFD")
		.replace(/[\u0300-\u036f]/g, "")
		.toLowerCase();
}

function pathStartsWith(root, candidate) {
	const relativePath = relative(resolve(root), resolve(candidate));
	return relativePath === "" || (
		relativePath !== ".." &&
		!relativePath.startsWith(`..${sep}`) &&
		!isAbsolute(relativePath)
	);
}

function relativePath(root, candidate) {
	return relative(resolve(root), resolve(candidate)).split(sep).join("/");
}

function pathMatchesDeclared(root, candidate, declaredPaths = []) {
	const candidateRelative = relativePath(root, candidate).toLowerCase();
	return declaredPaths.some((declared) => {
		const normalized = String(declared).replaceAll("\\", "/").replace(/^\.\//, "").toLowerCase();
		return candidateRelative === normalized || candidateRelative.startsWith(`${normalized}/`);
	});
}

function protectedPath(root, candidate) {
	const protectedRoots = [
		resolve(root, ".git"),
		resolve(root, ".senpi"),
		resolve(root, "context"),
		resolve(root, "workflow"),
		resolve(root, "var", "context"),
		// El verificador. Un agente que puede editar sus graders, sus umbrales o
		// la malla de evaluacion puede aprobar cualquier mecanismo cambiando la
		// vara en vez del resultado.
		resolve(root, "evals"),
		// Los claims aceptados los escribe el investigador, no un agente.
		resolve(root, "registry", "claims.jsonl"),
	];
	return protectedRoots.find((protectedRoot) => pathStartsWith(protectedRoot, candidate));
}

function roleName(role) {
	return String(role?.id ?? DEFAULT_ROLE);
}

export function canMutateCanonicalVault({ role, command }) {
	const normalizedCommand = String(command ?? "")
		.replaceAll("\\", "/")
		.toLowerCase();
	return roleName(role) === DEFAULT_ROLE
		&& /(?:^|[^a-z0-9_-])pikan-vault\/pikan-vault(?:\/|$)/.test(normalizedCommand);
}

function commandPathCandidates(command) {
	const text = String(command ?? "");
	const candidates = new Set();
	const quotedPattern = /["']((?:[A-Za-z]:[\\/]|\/)[^"']+)["']/g;
	const unquotedPattern = /(?:^|[\s(])((?:[A-Za-z]:[\\/]|\/)[^\s"';&|]+)/g;

	for (const match of text.matchAll(quotedPattern)) candidates.add(match[1]);
	for (const match of text.matchAll(unquotedPattern)) candidates.add(match[1]);

	return [...candidates].map((candidate) => candidate.replace(/[),.;]+$/, ""));
}

function externalCommandPaths({ root, command }) {
	const projectRoot = resolve(root);
	return commandPathCandidates(command)
		.map((candidate) => resolve(candidate))
		.filter((candidate) => !pathStartsWith(projectRoot, candidate));
}

function externalReadMessage(paths) {
	return "Aviso: lectura externa en modo solo lectura — " + paths.join(", ") + ".";
}

export function externalReadNotice({ root, command }) {
	const paths = externalCommandPaths({ root, command });
	return paths.length ? externalReadMessage(paths) : undefined;
}

export function externalMutationReason({ root, command }) {
	const paths = externalCommandPaths({ root, command });
	return paths.length
		? "Escritura externa bloqueada fuera de AF_PIKANs: " + paths.join(", ") + "."
		: undefined;
}

export function isContained(root, candidate) {
	return pathStartsWith(root, candidate);
}

export function parseResearchDirective(prompt, roles = []) {
	const roleIds = new Set(roles.map((role) => String(role?.id ?? "")));
	const defaultRole = roleIds.has(DEFAULT_ROLE) ? DEFAULT_ROLE : String(roles[0]?.id ?? DEFAULT_ROLE);
	const text = String(prompt ?? "");
	const match = text.match(/(?:research[-_ ]role|research\s+role)\s*[:=]\s*([A-Za-z0-9_-]+)/i)
		?? text.match(/(?:^|\s)\/research(?:\s+role)?\s+([A-Za-z0-9_-]+)/i);

	if (!match) return { role: defaultRole, explicit: false };
	const requested = match[1];
	if (roleIds.has(requested)) return { role: requested, explicit: true };
	return {
		role: defaultRole,
		explicit: true,
		error: `Unknown research role "${requested}". Available roles: ${[...roleIds].join(", ")}.`,
	};
}

function workflowItems(workflowContext) {
	return Array.isArray(workflowContext?.items) ? workflowContext.items : [];
}

function workflowMatches(item, normalizedPrompt) {
	const triggers = Array.isArray(item?.triggers) ? item.triggers : [];
	const excludes = Array.isArray(item?.excludes) ? item.excludes : [];
	const matchedTriggers = triggers.filter((trigger) => normalizedPrompt.includes(normalizeText(trigger)));
	const excluded = excludes.some((exclude) => normalizedPrompt.includes(normalizeText(exclude)));
	return { matchedTriggers, excluded };
}

function routeResult({ item, role, explicit, confidence, matchedTriggers = [], error }) {
	const workflow = item
		? { ...item, confidence, matchedTriggers }
		: {
			id: "literature",
			role: DEFAULT_ROLE,
			priority: 0,
			triggers: [],
			excludes: [],
			stages: ["orient"],
			gate_ids: [],
			confidence,
			matchedTriggers,
		};
	return {
		workflow,
		role: role ?? { id: DEFAULT_ROLE, purpose: "Conservative read-only research role.", read_paths: [], write_paths: [] },
		explicit,
		confidence,
		matchedTriggers,
		...(error ? { error } : {}),
	};
}

export function selectWorkflow(prompt, workflowContext = {}, roles = []) {
	const roleList = Array.isArray(roles) ? roles : [];
	const directive = parseResearchDirective(prompt, roleList);
	const items = workflowItems(workflowContext);
	const defaultItem = items.find((item) => item?.id === workflowContext?.default)
		?? items.find((item) => item?.role === DEFAULT_ROLE)
		?? items[0];
	const defaultRole = roleList.find((role) => role?.id === DEFAULT_ROLE)
		?? roleList[0]
		?? { id: DEFAULT_ROLE, purpose: "Conservative read-only research role.", read_paths: [], write_paths: [] };

	if (directive.explicit) {
		const item = items.find((candidate) => candidate?.role === directive.role) ?? defaultItem;
		const role = roleList.find((candidate) => candidate?.id === directive.role) ?? defaultRole;
		return routeResult({
			item,
			role,
			explicit: true,
			confidence: directive.error ? "fallback" : "explicit",
			error: directive.error,
		});
	}

	const normalizedPrompt = normalizeText(prompt);
	const ranked = items
		.map((item, index) => {
			const match = workflowMatches(item, normalizedPrompt);
			return {
				item,
				index,
				matchedTriggers: match.matchedTriggers,
				score: match.excluded ? 0 : match.matchedTriggers.length,
			};
		})
		.filter((candidate) => candidate.score > 0)
		.sort((left, right) =>
			right.score - left.score
			|| Number(right.item?.priority ?? 0) - Number(left.item?.priority ?? 0)
			|| left.index - right.index);

	if (!ranked.length) {
		const role = roleList.find((candidate) => candidate?.id === defaultItem?.role) ?? defaultRole;
		return routeResult({ item: defaultItem, role, explicit: false, confidence: "fallback" });
	}

	const winner = ranked[0];
	const runnerUp = ranked[1];
	const unresolvedTie = runnerUp
		&& runnerUp.score === winner.score
		&& Number(runnerUp.item?.priority ?? 0) === Number(winner.item?.priority ?? 0);
	if (unresolvedTie) {
		const role = roleList.find((candidate) => candidate?.id === defaultItem?.role) ?? defaultRole;
		return routeResult({ item: defaultItem, role, explicit: false, confidence: "fallback" });
	}
	const role = roleList.find((candidate) => candidate?.id === winner.item?.role);
	if (!role) {
		return routeResult({
			item: defaultItem,
			role: defaultRole,
			explicit: false,
			confidence: "fallback",
			error: `Workflow "${winner.item?.id ?? "unknown"}" references an unavailable role.`,
		});
	}
	return routeResult({
		item: winner.item,
		role,
		explicit: false,
		confidence: winner.score > 1 ? "high" : "medium",
		matchedTriggers: winner.matchedTriggers,
	});
}

export function filesystemDecision({ root, role, operation, canonicalPath }) {
	const candidate = resolve(canonicalPath);
	const projectRoot = resolve(root);
	if (!isContained(projectRoot, candidate)) {
		if (operation === "read") {
			return {
				allow: true,
				notice: externalReadMessage([candidate]),
			};
		}
		return { allow: false, reason: "Access is limited to the AF_PIKANs repository." };
	}

	const protectedRoot = protectedPath(projectRoot, candidate);
	if (protectedRoot && (operation === "write" || protectedRoot.endsWith(`${sep}.git`) || protectedRoot.endsWith(`${sep}.senpi`))) {
		return {
			allow: false,
			reason: "Canonical contracts, generated context, and Senpi control files are protected.",
		};
	}

	if (operation === "write") {
		if (protectedRoot) {
			return { allow: false, reason: "The requested path is a protected research-control path." };
		}
		if (!pathMatchesDeclared(projectRoot, candidate, role?.write_paths)) {
			return {
				allow: false,
				reason: `Role ${roleName(role)} cannot write outside its declared surface.`,
			};
		}
		return { allow: true };
	}

	if (pathMatchesDeclared(projectRoot, candidate, ["var/context", "workflow"])) return { allow: true };
	if (!pathMatchesDeclared(projectRoot, candidate, role?.read_paths)) {
		return {
			allow: false,
			reason: `Role ${roleName(role)} cannot read or enumerate this path.`,
		};
	}
	return { allow: true };
}

function nodeText(node) {
	const values = [node?.id, node?.type, node?.kind, node?.title, node?.path, node?.role, node?.tags];
	return normalizeText(values.flatMap((value) => Array.isArray(value) ? value : [value]).join(" "));
}

function nodeScore(node, promptTokens) {
	const text = nodeText(node);
	const matches = promptTokens.reduce((score, token) => score + (text.includes(token) ? 1 : 0), 0);
	const typeBonus = {
		problem: 3,
		hypothesis: 2,
		model: 2,
		notebook: 1,
		experiment: 1,
	};
	return matches + (typeBonus[node?.type] ?? 0);
}

export function selectRelevantNodes(graph, prompt, limit = 12) {
	const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
	const promptTokens = normalizeText(prompt).match(/[a-z0-9]{3,}/g) ?? [];
	const scored = nodes.map((node, index) => ({ node, index, score: nodeScore(node, promptTokens) }));
	const matching = scored
		.filter((item) => item.score > 0)
		.sort((left, right) => right.score - left.score || left.index - right.index);
	const fallback = scored
		.filter((item) => item.score === 0)
		.sort((left, right) => left.index - right.index);
	return [...matching, ...fallback].slice(0, Math.max(0, limit)).map((item) => item.node);
}

function list(values) {
	return Array.isArray(values) && values.length ? values.map((value) => `\`${value}\``).join(", ") : "ninguno";
}

function graphSummary(graph) {
	const diagnostics = graph?.diagnostics ?? {};
	const audit = graph?.notebook_audit ?? {};
	return [
		`nodos=${graph?.nodes?.length ?? 0}`,
		`relaciones=${graph?.edges?.length ?? 0}`,
		`enlaces_faltantes=${diagnostics.missing_links?.length ?? 0}`,
		`enlaces_ambiguos=${diagnostics.ambiguous_links?.length ?? 0}`,
		`notebooks=${audit.count ?? 0}`,
	].join("; ");
}

export function renderSystemPrompt({ pack, graph, role, workflow, prompt, ready, error }) {
	const relevant = selectRelevantNodes(graph, prompt, 12);
	const lines = [
		"## PIKAN research automation context",
		"Este repositorio usa la ontología `pikan-research` y el nivel de autonomía `L2_human_verified_ai_execution`.",
		`Rol activo: **${roleName(role)}** — ${role?.purpose ?? "rol no descrito"}`,
		`Lectura permitida: ${list(role?.read_paths)}`,
		`Escritura permitida: ${list(role?.write_paths)}`,
		`Gates que requieren decisión humana: ${list(role?.approval_required)}`,
		...(workflow ? [
			`Workflow activo: \`${workflow.id ?? "literature"}\` — confidence \`${workflow.confidence ?? "unknown"}\``,
			`Fases: ${list(workflow.stages)}`,
			`Gates del workflow: ${list(workflow.gate_ids)}`,
			`Triggers detectados: ${list(workflow.matchedTriggers)}`,
		] : []),
		`Estado del grafo: ${graphSummary(graph)}`,
		"",
		"Reglas operativas:",
		"- Un agente propone; la aprobación humana decide claims, promoción de conocimiento, cambios de ontología y cambios del modelo/PDE.",
		"- No modificar `context/`, `workflow/`, `.senpi/`, `.git/` ni `var/context/`.",
		"- `evals/` es el verificador y es de solo lectura para todo rol: graders, umbrales, baselines congelados y la malla de evaluación no se negocian. Si un resultado exige cambiar la vara, detente y pide el gate `eval_integrity`.",
		"- Un mecanismo en `src/mechanisms/` no puede importar `evals`: un candidato que conoce su métrica la optimiza en vez del problema.",
		"- Tier 0 y Tier 1 se ejecutan libremente. Tier 2 requiere presupuesto aprobado; Tier 3 requiere el gate `scientific_claim`.",
		"- La promoción de notas a `PIKAN-Vault/PIKAN-Vault` requiere aprobación humana explícita y solo está declarada para `literature_scout`.",
		"- Se permiten lecturas externas en modo solo lectura; antes de cada una informa al usuario la ruta exacta y el motivo. Las escrituras externas están bloqueadas.",
		"- Si la tarea requiere otro rol o una ruta no declarada, detente y solicita la decisión humana.",
		"- Una nota bibliográfica respalda motivación o método; no sustituye evidencia experimental.",
		"",
		"Nodos relevantes del grafo:",
	];
	if (relevant.length) {
		lines.push(...relevant.map((node) => `- ${node.id ?? node.title ?? "sin-id"} — ${node.title ?? node.path ?? "sin título"}`));
	} else {
		lines.push("- No hay nodos disponibles.");
	}

	if (!ready) {
		lines.unshift(
			"### Context unavailable",
			`No ejecutes cambios ni corridas hasta corregir el contexto de investigación. Motivo: ${error ?? "validación no disponible"}`,
			"",
		);
	}
	return lines.join("\n");
}
