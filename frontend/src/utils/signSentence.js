const EXACT_SENTENCES = {
  xin_chao: "Xin chào.",
  cam_on: "Cảm ơn.",

  "toi yeu ban": "Tôi yêu bạn.",
  "toi nha ve": "Tôi về nhà.",
  "toi truong di": "Tôi đi đến trường.",
  "toi an muon": "Tôi muốn ăn.",
  "toi uong muon": "Tôi muốn uống.",
  "toi giup_do can": "Tôi cần giúp đỡ.",
};


const LABEL_TEXT = {
  xin_chao: "Xin chào",
  giup_do: "giúp đỡ",
  cam_on: "Cảm ơn",

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
  const cleanedLabels = labels.filter(
    (label) =>
      Boolean(label) &&
      label !== "no_sign",
  );


  if (cleanedLabels.length === 0) {
    return "";
  }


  const key = cleanedLabels.join(" ");


  if (EXACT_SENTENCES[key]) {
    return EXACT_SENTENCES[key];
  }


  const words = cleanedLabels
    .map((label) => LABEL_TEXT[label] ?? label)
    .filter(Boolean);


  if (words.length === 0) {
    return "";
  }


  const text = words.join(" ");


  return (
    text.charAt(0).toUpperCase() +
    text.slice(1)
  );
}