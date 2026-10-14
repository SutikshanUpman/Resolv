import { docText } from './fact';

export interface TeamMember {
  readonly name: string;
  /** Optional; fill in when known. Nothing is shown for an empty field. */
  readonly role?: string;
  readonly github?: string;
  readonly linkedin?: string;
}

// Title block of Documentation_template.md
export const teamName = docText('The Epoch Warriors', 'title');
export const membersText = docText('Rajveer Gupta, Manas Tiwari, Hrutuparna Bedekar, Sutikshan Upman', 'title');

export const team: readonly TeamMember[] = [
  { name: 'Rajveer Gupta' },
  { name: 'Manas Tiwari' },
  { name: 'Hrutuparna Bedekar' },
  { name: 'Sutikshan Upman' },
];
