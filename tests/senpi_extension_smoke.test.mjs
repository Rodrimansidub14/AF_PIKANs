import test from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";

import extension from "../.senpi/extensions/pikan-research-context.ts";

test("registers the Senpi research context integration", () => {
	const registrations = {
		commands: [],
		events: [],
		policies: [],
	};
	const pi = {
		registerCommand(name, options) {
			registrations.commands.push({ name, options });
		},
		registerFilesystemPolicy(policy) {
			registrations.policies.push(policy);
		},
		on(name, handler) {
			registrations.events.push({ name, handler });
		},
	};

	extension(pi);

	assert.deepEqual(registrations.commands.map((item) => item.name), ["research-context"]);
	assert.deepEqual(
		registrations.events.map((item) => item.name).sort(),
		["before_agent_start", "session_start", "tool_call"].sort(),
	);
	assert.equal(registrations.policies.length, 1);
	assert.equal(typeof registrations.policies[0].check, "function");
});

test("injects the validated project context before an agent turn", async () => {
	const registrations = { events: [], policies: [] };
	const pi = {
		registerCommand() {},
		registerFilesystemPolicy(policy) {
			registrations.policies.push(policy);
		},
		on(name, handler) {
			registrations.events.push({ name, handler });
		},
	};

	extension(pi);
	const beforeAgentStart = registrations.events.find((item) => item.name === "before_agent_start").handler;
	const result = await beforeAgentStart(
		{
			prompt: "adapta el baseline Heat en el notebook",
			systemPrompt: "Base Senpi system prompt",
		},
		{
			ui: { notify() {}, setStatus() {} },
		},
	);

	assert.match(result.systemPrompt, /Base Senpi system prompt/);
	assert.match(result.systemPrompt, /PIKAN research automation context/);
	assert.match(result.systemPrompt, /notebook_engineer/);
	assert.match(result.systemPrompt, /Workflow activo:.*notebook/);
	assert.match(result.systemPrompt, /nodos=\d+/);

	const policy = registrations.policies[0];
	assert.deepEqual(
		policy.check({ operation: "write", canonicalPath: join(process.cwd(), "notebooks", "candidate.ipynb") }),
		{ allow: true },
	);
	assert.equal(
		policy.check({ operation: "write", canonicalPath: join(process.cwd(), "context", "research-ontology.toml") }).allow,
		false,
	);
	assert.equal(
		policy.check({ operation: "read", canonicalPath: join(process.cwd(), "workflow", "index.toml") }).allow,
		true,
	);
	assert.equal(
		policy.check({ operation: "write", canonicalPath: join(process.cwd(), "workflow", "index.toml") }).allow,
		false,
	);
	assert.equal(
		policy.check({ operation: "write", canonicalPath: join(process.cwd(), "..", "IMP", "obs-data", "anything.md") }).allow,
		false,
	);
	assert.equal(
		policy.check({ operation: "read", canonicalPath: join(process.cwd(), "..", "Downloads", "source.md") }).allow,
		true,
	);
	assert.equal(
		policy.check({ operation: "write", canonicalPath: join(process.cwd(), "..", "Downloads", "source.md") }).allow,
		false,
	);
});

test("notifies external shell reads and blocks external shell writes", async () => {
	const registrations = { events: [], policies: [] };
	const notifications = [];
	const pi = {
		registerCommand() {},
		registerFilesystemPolicy(policy) {
			registrations.policies.push(policy);
		},
		on(name, handler) {
			registrations.events.push({ name, handler });
		},
	};

	extension(pi);
	const toolCall = registrations.events.find((item) => item.name === "tool_call").handler;
	const externalPath = join(process.cwd(), "..", "Downloads", "source.md");
	const context = {
		ui: {
			notify(message, type) {
				notifications.push({ message, type });
			},
		},
	};

	const readResult = await toolCall({
		toolName: "bash",
		input: { command: `Get-Content "${externalPath}"` },
	}, context);
	assert.equal(readResult, undefined);
	assert.match(notifications[0].message, /lectura externa/i);
	assert.match(notifications[0].message, /source\.md/i);

	const writeResult = await toolCall({
		toolName: "bash",
		input: { command: `Copy-Item "${externalPath}" "${join(process.cwd(), "docs", "source.md")}"` },
	}, context);
	assert.equal(writeResult.block, true);
	assert.match(writeResult.reason, /escritura externa/i);
});
