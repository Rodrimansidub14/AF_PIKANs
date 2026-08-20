import test from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";

import {
	filesystemDecision,
	isContained,
	parseResearchDirective,
	renderSystemPrompt,
	selectWorkflow,
	selectRelevantNodes,
} from "../.senpi/extensions/pikan-research-context-core.mjs";

const ROOT = process.platform === "win32" ? "C:\\repo" : "/repo";
const PACK = {
	ontology: "pikan-research",
	autonomy_level: "L2_human_verified_ai_execution",
	paths: {
		vault: "PIKAN-Vault/PIKAN-Vault",
		notebooks: "notebooks",
		generated_output: "var/context",
	},
	roles: [
		{
			id: "literature_scout",
			purpose: "Leer literatura y proponer evidencia.",
			read_paths: ["PIKAN-Vault/PIKAN-Vault", "context", "docs"],
			write_paths: ["var/agent_runs"],
			approval_required: ["modify_vault_note", "modify_ontology"],
		},
		{
			id: "notebook_engineer",
			purpose: "Editar notebooks acotados.",
			read_paths: ["notebooks", "src", "context", "PIKAN-Vault/PIKAN-Vault"],
			write_paths: ["var/agent_runs", "notebooks"],
			approval_required: ["modify_core_model", "promote_notebook"],
		},
		{
			id: "evaluator",
			purpose: "Comparar corridas y calcular gates.",
			read_paths: ["var", "artifacts", "context"],
			write_paths: ["var/agent_runs"],
			approval_required: ["accept_result", "promote_claim"],
		},
	],
	gates: [
		{
			id: "scope",
			required_before: ["edit_notebook"],
			decision_owner: "human",
		},
	],
	graph: { nodes: 4, edges: 3, missing_links: 0, ambiguous_links: 0 },
	notebooks: { count: 1, status_counts: { unclassified: 1 } },
	workflow: {
		default: "literature",
		ambiguity_policy: "conservative",
		items: [
			{
				id: "literature",
				role: "literature_scout",
				priority: 10,
				triggers: ["literatura", "fuentes", "marco teorico", "resume"],
				excludes: ["notebook"],
				stages: ["orient", "extract_evidence"],
				gate_ids: ["scientific_claim"],
			},
			{
				id: "hypothesis",
				role: "hypothesis_planner",
				priority: 20,
				triggers: ["hipotesis", "falsable", "manifiesto"],
				excludes: [],
				stages: ["define_question", "draft_hypothesis"],
				gate_ids: ["scope", "hypothesis"],
			},
			{
				id: "notebook",
				role: "notebook_engineer",
				priority: 30,
				triggers: ["notebook", "adapta", "baseline"],
				excludes: ["solo explica"],
				stages: ["inspect_contract", "edit_declared_surface", "audit"],
				gate_ids: ["scope", "code_surface"],
			},
			{
				id: "run",
				role: "experiment_runner",
				priority: 40,
				triggers: ["ejecuta", "corrida", "run-local"],
				excludes: [],
				stages: ["validate_manifest", "run_local"],
				gate_ids: ["hypothesis"],
			},
			{
				id: "evaluation",
				role: "evaluator",
				priority: 35,
				triggers: ["metricas", "compara", "resultado"],
				excludes: [],
				stages: ["read_runs", "compute_metrics"],
				gate_ids: ["scientific_claim"],
			},
			{
				id: "synthesis",
				role: "research_synthesizer",
				priority: 25,
				triggers: ["redacta", "latex", "sintetiza"],
				excludes: [],
				stages: ["read_evidence", "draft_report"],
				gate_ids: ["scientific_claim"],
			},
		],
	},
};

test("defaults to conservative literature_scout role", () => {
	assert.deepEqual(parseResearchDirective("resume el estado del vault", PACK.roles), {
		role: "literature_scout",
		explicit: false,
	});
});

test("parses an explicit research role", () => {
	assert.deepEqual(
		parseResearchDirective("research-role=notebook_engineer adapta el notebook", PACK.roles),
		{ role: "notebook_engineer", explicit: true },
	);
});

test("reports invalid explicit research role", () => {
	const result = parseResearchDirective("research-role=unknown crea evidencia", PACK.roles);

	assert.equal(result.role, "literature_scout");
	assert.equal(result.explicit, true);
	assert.match(result.error, /unknown/i);
});

test("checks path containment without allowing sibling prefixes", () => {
	assert.equal(isContained(ROOT, join(ROOT, "notebooks", "baseline.ipynb")), true);
	assert.equal(isContained(ROOT, `${ROOT}-backup`), false);
	assert.equal(isContained(ROOT, join(ROOT, "..", "obs-data")), false);
});

test("allows declared notebook reads and writes for notebook_engineer", () => {
	assert.deepEqual(
		filesystemDecision({
			root: ROOT,
			role: PACK.roles[1],
			operation: "read",
			canonicalPath: join(ROOT, "notebooks", "baseline.ipynb"),
		}),
		{ allow: true },
	);
	assert.deepEqual(
		filesystemDecision({
			root: ROOT,
			role: PACK.roles[1],
			operation: "write",
			canonicalPath: join(ROOT, "notebooks", "baseline.ipynb"),
		}),
		{ allow: true },
	);
});

test("denies outside-root and protected writes", () => {
	const role = PACK.roles[1];
	const cases = [
		join(ROOT, "..", "obs-data", "secret.md"),
		join(ROOT, "context", "research-ontology.toml"),
		join(ROOT, "PIKAN-Vault", "PIKAN-Vault", "Paper.md"),
		join(ROOT, "var", "context", "research-graph.json"),
	];

	for (const canonicalPath of cases) {
		const result = filesystemDecision({ root: ROOT, role, operation: "write", canonicalPath });
		assert.equal(result.allow, false, canonicalPath);
	}
});

test("denies a role from writing outside its declared surface", () => {
	const result = filesystemDecision({
		root: ROOT,
		role: PACK.roles[0],
		operation: "write",
		canonicalPath: join(ROOT, "notebooks", "new.ipynb"),
	});

	assert.equal(result.allow, false);
	assert.match(result.reason, /literature_scout|write|surface/i);
});

test("allows every role to read generated context", () => {
	const result = filesystemDecision({
		root: ROOT,
		role: PACK.roles[0],
		operation: "read",
		canonicalPath: join(ROOT, "var", "context", "senpi-context.json"),
	});

	assert.deepEqual(result, { allow: true });
});

test("selects graph nodes relevant to the prompt", () => {
	const graph = {
		nodes: [
			{ id: "vault:PINN", type: "concept", title: "PINN", path: "concepts/PINN.md" },
			{ id: "vault:Heat", type: "problem", title: "Heat equation", path: "domains/Heat.md" },
			{ id: "notebook:baseline", type: "notebook", title: "baseline_heat", path: "notebooks/baseline_heat.ipynb" },
		],
	};

	const selected = selectRelevantNodes(graph, "baseline notebook para heat equation", 2);

	assert.equal(selected.length, 2);
	assert.deepEqual(
		selected.map((node) => node.id),
		["vault:Heat", "notebook:baseline"],
	);
});

test("infers notebook workflow from a natural-language prompt", () => {
	const result = selectWorkflow("adapta el baseline Heat en el notebook", PACK.workflow, PACK.roles);

	assert.equal(result.workflow.id, "notebook");
	assert.equal(result.role.id, "notebook_engineer");
	assert.equal(result.explicit, false);
	assert.deepEqual(result.matchedTriggers.sort(), ["adapta", "baseline", "notebook"].sort());
});

test("falls back conservatively when no workflow trigger matches", () => {
	const result = selectWorkflow("revisa este asunto general", PACK.workflow, PACK.roles);

	assert.equal(result.workflow.id, "literature");
	assert.equal(result.role.id, "literature_scout");
	assert.equal(result.confidence, "fallback");
});

test("explicit role remains an override over inferred workflow", () => {
	const result = selectWorkflow("research-role=evaluator adapta el notebook", PACK.workflow, PACK.roles);

	assert.equal(result.workflow.id, "evaluation");
	assert.equal(result.role.id, "evaluator");
	assert.equal(result.explicit, true);
});

test("rendered context includes the active workflow and phases", () => {
	const rendered = renderSystemPrompt({
		pack: PACK,
		graph: PACK,
		role: PACK.roles[1],
		workflow: PACK.workflow.items[2],
		prompt: "adapta el baseline",
		ready: true,
	});

	assert.match(rendered, /Workflow activo:.*notebook/);
	assert.match(rendered, /inspect_contract/);
});

test("renders an actionable system context and fail-closed notice", () => {
	const ready = renderSystemPrompt({
		pack: PACK,
		graph: { ...PACK, nodes: [{ id: "vault:PINN", type: "concept", title: "PINN", path: "concepts/PINN.md" }] },
		role: PACK.roles[0],
		prompt: "resume PINN",
		ready: true,
	});
	assert.match(ready, /pikan-research/);
	assert.match(ready, /literature_scout/);
	assert.match(ready, /PINN/);
	assert.doesNotMatch(ready, /Context unavailable/);

	const blocked = renderSystemPrompt({
		pack: PACK,
		graph: PACK,
		role: PACK.roles[0],
		prompt: "cualquier tarea",
		ready: false,
		error: "contrato inválido",
	});
	assert.match(blocked, /Context unavailable/);
	assert.match(blocked, /contrato inválido/);
});
