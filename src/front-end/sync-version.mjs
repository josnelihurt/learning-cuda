import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";

const frontendRoot = resolve(process.cwd());
const versionPath = resolve(frontendRoot, "VERSION");
const packageJsonPath = resolve(frontendRoot, "package.json");
const packageLockPath = resolve(frontendRoot, "package-lock.json");

const canonicalVersion = readFileSync(versionPath, "utf8").trim();

if (!canonicalVersion) {
  console.error("VERSION file is empty");
  process.exit(1);
}

const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
let hasChanges = false;

if (packageJson.version !== canonicalVersion) {
  packageJson.version = canonicalVersion;
  hasChanges = true;
  writeFileSync(packageJsonPath, `${JSON.stringify(packageJson, null, 2)}\n`);
}

const packageLock = JSON.parse(readFileSync(packageLockPath, "utf8"));
let lockChanged = false;

if (packageLock.version !== canonicalVersion) {
  packageLock.version = canonicalVersion;
  lockChanged = true;
}

if (packageLock.packages?.[""]?.version !== canonicalVersion) {
  packageLock.packages[""].version = canonicalVersion;
  lockChanged = true;
}

if (lockChanged) {
  hasChanges = true;
  writeFileSync(packageLockPath, `${JSON.stringify(packageLock, null, 2)}\n`);
}

if (hasChanges) {
  console.log(`Synchronized frontend version to ${canonicalVersion}`);
} else {
  console.log(`Frontend version already synchronized (${canonicalVersion})`);
}
