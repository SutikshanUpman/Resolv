import type { Split } from '../../content/fact';
import { splitLabels } from '../../content/labels';
import { Chip } from './Chip';

/** Names the data a number was measured on, next to the number. */
export function SplitTag({ split }: { split?: Split }) {
  if (!split) return null;
  return (
    <Chip tone={split === 'leaderboard' ? 'accent' : 'neutral'} title={splitLabels[split].long}>
      {splitLabels[split].short}
    </Chip>
  );
}
