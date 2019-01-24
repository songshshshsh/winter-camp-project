interface TextArgs {
  text: string;
}

interface TextRes {
  info?: string
  label: string;
}

interface RunArgs {
  text: string;
  label0: string;
  label1: string;
}

interface RunRes {
  info?: string;
  res: string;
}
