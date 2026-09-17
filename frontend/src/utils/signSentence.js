const EXACT_SENTENCES = {
  "toi an muon": "Tôi muốn ăn.",
  "toi uong muon": "Tôi muốn uống.",
  "toi yeu ban": "Tôi yêu bạn.",
  "toi nha ve": "Tôi về nhà.",
  "toi truong di": "Tôi đi đến trường.",
  "toi an muon": "Tôi muốn ăn.",
  "toi uong muon": "Tôi muốn uống.",
  "toi giup_do can": "Tôi cần giúp đỡ.",
};


const LABEL_TEXT = {
  xin_chao: "xin chào",
  giup_do: "giúp đỡ",
  cam_on: "cảm ơn",

  co: "có",
  cong_nghe: "công nghệ",

  khong: "không",
  khong_cho: "không cho",
  khong_biet: "không biết",

  can: "cần",
  cho_1: "chợ",
  day: "dạy",

  muon: "muốn",
  di: "đi",
  uong: "uống",
  an: "ăn",

  yeu: "yêu",

  ban: "bạn",
  ban_1: "bận",

  buon: "buồn",
  biet: "biết",

  toi: "tôi",
  thich: "thích",

  hoc: "học",
  lam: "làm",
  nha: "nhà",
  truong: "trường",

  sieu_thi: "siêu thị",

  ve: "về",
  vui: "vui",
  den: "đến",

  no_sign: "",
};


export function translateSignLabels(labels) {
  if (!Array.isArray(labels)) {
    return "";
  }

  const cleanedLabels = labels
    .filter(Boolean)
    .map((label) => String(label).trim().toLowerCase())
    .filter((label) => label && label !== "no_sign");

  if (cleanedLabels.length === 0) {
    return "";
  }

  const key = cleanedLabels.join(" ");

  // 1. Câu mẫu chính xác
  if (EXACT_SENTENCES[key]) {
    return EXACT_SENTENCES[key];
  }

  // 2. Luật NLP: Tôi + yêu + bạn
  if (
    cleanedLabels.includes("toi") &&
    cleanedLabels.includes("yeu") &&
    cleanedLabels.includes("ban")
  ) {
    return "Tôi yêu bạn.";
  }

  // 3. Luật NLP: Tôi + muốn + ăn
  if (
    cleanedLabels.includes("toi") &&
    cleanedLabels.includes("muon") &&
    cleanedLabels.includes("an")
  ) {
    return "Tôi muốn ăn.";
  }

  // 4. Luật NLP: Tôi + muốn + uống
  if (
    cleanedLabels.includes("toi") &&
    cleanedLabels.includes("muon") &&
    cleanedLabels.includes("uong")
  ) {
    return "Tôi muốn uống.";
  }

  // 5. Luật NLP: Tôi + cần + giúp đỡ
  if (
    cleanedLabels.includes("toi") &&
    cleanedLabels.includes("can") &&
    cleanedLabels.includes("giup_do")
  ) {
    return "Tôi cần giúp đỡ.";
  }

  // 6. Nếu không có mẫu câu → ghép từ
  const words = cleanedLabels
    .map((label) => LABEL_TEXT[label] ?? label)
    .filter(Boolean);

  if (words.length === 0) {
    return "";
  }

  const text = words.join(" ");

  return (
    text.charAt(0).toUpperCase() +
    text.slice(1) +
    "."
  );
}