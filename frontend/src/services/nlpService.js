import { translateSignLabels } from "../utils/signSentence";

export function processSignLabels(labels) {
  if (!Array.isArray(labels)) {
    return "";
  }

  const normalizedLabels = labels
    .filter(Boolean)
    .map((label) => String(label).trim().toLowerCase())
    .filter((label) => label && label !== "no_sign");

  if (normalizedLabels.length === 0) {
    return "";
  }

  return translateSignLabels(normalizedLabels);
}