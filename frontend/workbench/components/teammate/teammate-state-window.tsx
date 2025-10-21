import { useWebSocket } from '@/context/socket';
import { TeamMemberState, ValidTeamMemberStatus } from '@/lib/types';
import Avatar from '@mui/material/Avatar';
import Popover from '@mui/material/Popover';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import html2canvas from 'html2canvas';
import { MouseEvent, useState } from 'react';

export interface TeammateStateWindowProps {
  envId: string;
  teammateState: TeamMemberState;
}

export function TeammateStateWindow({ envId, teammateState }: TeammateStateWindowProps) {
  const { connect, disconnect, state, isConnected, error } = useWebSocket();
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false);
  const [feedbackType, setFeedbackType] = useState<string | null>(null);
  const [feedbackText, setFeedbackText] = useState('');
  const [screenshot, setScreenshot] = useState<string | null>(null);

  const handleAvatarClick = (event: MouseEvent<HTMLElement>): void => {
    setAnchorEl(event.currentTarget);
  };

  const handlePopoverClose = (): void => {
    setAnchorEl(null);
  };

  const handleThumbClick = async (type: 'thumb-up' | 'thumb-down'): Promise<void> => {
    setFeedbackType(type);
    const canvas = await html2canvas(document.body);
    const image = canvas.toDataURL('image/png');
    setScreenshot(image);
    setFeedbackModalOpen(true);
  };

  const handleModalClose = (): void => {
    setFeedbackModalOpen(false);
    setFeedbackText('');
    setScreenshot(null);
  };

  const open = Boolean(anchorEl);
  const id = open ? 'agent-popover' : undefined;

  return (
    <div className="flex items-center space-x-4 h-1/9">
      <div className="relative h-full flex items-center" style={{ width: 'auto', aspectRatio: "1 / 1" }}>
        <Tooltip title="Click to see agent state details.">
          <Avatar
            src="/avatar.svg"
            sx={{
              height: '75%',
              width: 'auto',
              aspectRatio: "1 / 1",
              backgroundColor: 'white',
              ml: 2,
              cursor: 'pointer',
              transition: 'transform 0.2s, opacity 0.2s',
              '&:hover': {
                transform: 'scale(1.05)',
                opacity: 0.8,
              },
            }}
            onClick={handleAvatarClick}
          />
        </Tooltip>
        {teammateState.status === ValidTeamMemberStatus.WORKING &&
          teammateState.action.length > 0 && (
            <div className="absolute bottom-3 right-0 w-3 h-3 bg-green-500 rounded-full" />
          )}
        {(teammateState.status === ValidTeamMemberStatus.IDLE ||
          teammateState.action.length === 0) && (
          <div className="absolute bottom-3 right-0 w-3 h-3 bg-yellow-500 rounded-full" />
        )}
        {teammateState.status === ValidTeamMemberStatus.FAILED && (
          <div className="absolute bottom-3 right-0 w-3 h-3 bg-red-500 rounded-full" />
        )}
      </div>
      <div className="flex flex-row justify-center">
        <div className="mr-8">
          <p className="text-sm">
            <span className="inline-block w-3 h-3 rounded-full mr-2 bg-green-500" />
            Working
          </p>
          <p className="text-sm">
            <span className="inline-block w-3 h-3 rounded-full mr-2 bg-yellow-500" />
            Idle
          </p>
          <p className="text-sm">
            <span className="inline-block w-3 h-3 rounded-full mr-2 bg-red-500" />
            Failed
          </p>
        </div>
      </div>
      <Popover
        id={id}
        open={open}
        anchorEl={anchorEl}
        onClose={handlePopoverClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'center',
        }}>
        <Typography sx={{ p: 2 }}>
          {teammateState.status === ValidTeamMemberStatus.WORKING &&
            teammateState.action.length > 0 && <span>{teammateState.action}</span>}
          {(teammateState.status === ValidTeamMemberStatus.IDLE ||
            teammateState.action.length === 0) && <span>The agent is currently idle.</span>}
          {teammateState.status === ValidTeamMemberStatus.FAILED && (
            <span>The agent has failed to complete the task.</span>
          )}
        </Typography>
      </Popover>
    </div>
  );
}
